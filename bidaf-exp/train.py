#!python3
"""
Training script: load a config file, create a new model using it,
then train that model.
"""
import sys
import json
import yaml
import os.path
import itertools
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import h5py

import checkpointing
from Mydataset import load_data, tokenize_data, gen_vocab, EpochGen
from Mydataset import SymbolEmbSourceNorm
from Mydataset import SymbolEmbSourceText
from Mydataset import symbol_injection


def try_to_resume(force_restart, exp_folder, checkpointfile):
    if force_restart:
        return None, None, 0
    elif os.path.isfile(exp_folder + checkpointfile):
        checkpoint = h5py.File(exp_folder + checkpointfile)
        epoch = checkpoint['training/epoch'][()] + 1
        # Try to load training state.
        try:
            training_state = torch.load(exp_folder + '/'+checkpointfile+'.opt')
        except FileNotFoundError:
            training_state = None
    else:# can not find check point
        return None, None, 0

    return checkpoint, training_state, epoch


def reload_state(checkpoint, training_state, config, args):
    """
    Reload state when resuming training.
    """
    model, id_to_token, id_to_char = MyModel.from_checkpoint(
        config['bidaf'], checkpoint)
    if torch.cuda.is_available() and args['cuda']:
        model.cuda()
    model.train()

    optimizer = get_optimizer(model, config, training_state)

    token_to_id = {tok: id_ for id_, tok in id_to_token.items()}
    char_to_id = {char: id_ for id_, char in id_to_char.items()}

    len_tok_voc = len(token_to_id)
    len_char_voc = len(char_to_id)

    with open(args['data']) as f_o:
        data, _ = load_data(json.load(f_o),
                            span_only=True, answered_only=True)
    limit_passage = config.get('training', {}).get('limit')
    data = tokenize_data(data, token_to_id, char_to_id, limit_passage)

    data = get_loader(data, config)

    assert len(token_to_id) == len_tok_voc
    assert len(char_to_id) == len_char_voc

    return model, id_to_token, id_to_char, optimizer, data


def get_optimizer(model, config, state):
    """
    Get the optimizer
    """
    parameters = filter(lambda p: p.requires_grad,
                        model.parameters())
    optimizer = optim.Adam(
        parameters,
        lr=config['training'].get('lr', 0.01),
        betas=config['training'].get('betas', (0.9, 0.999)),
        eps=config['training'].get('eps', 1e-8),
        weight_decay=config['training'].get('weight_decay', 0))

    if state is not None:
        optimizer.load_state_dict(state)

    return optimizer


def get_loader(data, vocab, cvocab, config):
    data = EpochGen(
        data, vocab, cvocab,
        batch_size=config.get('training', {}).get('batch_size', 32),
        shuffle=config.get('training',{}).get('shuffle',True),
        sample=config.get('training',{}).get('sample', False) )
    return data

def init_state(config, args):
    '''
    initialize state, loading from preprocessed data
    '''
    print('Loading data...')
    # data = json.load(open(args['exp_folder']+'/debug/debug_train_data.json', 'r'))
    data = json.load(open(args['exp_folder'] + args['data'], 'r'))
    id_to_token = json.load(open(args['exp_folder']+'/preprocessed/vocab.json', 'r'))
    id_to_char = json.load(open(args['exp_folder']+'/preprocessed/cvocab.json','r'))
    print('len of vocab: ', len(id_to_token))
    print('len of cvocab: ', len(id_to_char))
    token_to_id = {tok_ : int(id_) for id_, tok_ in id_to_token.items()}
    char_to_id = {char_: int(id_) for id_, char_ in id_to_char.items()}
    print('Get loader...')
    data = get_loader(data, token_to_id, char_to_id, config)
    print('Creating model...')
    model = MyModel.from_config(config['bidaf'], id_to_token, id_to_char, args['cuda'])

    if args['word_rep']:
        print('Loading pre-trained embeddings...')
        with open(args['word_rep']) as f_o:
            pre_trained = SymbolEmbSourceText(
                    f_o,
                    set(tok for id_, tok in id_to_token.items() if int(id_) != 0))
        mean, cov = pre_trained.get_norm_stats(args['use_covariance'])
        rng = np.random.RandomState(2)
        oovs = SymbolEmbSourceNorm(mean, cov, rng, args['use_covariance'])
        
        print('Initialize model embeddings...')
        model.embedder.embeddings[0].embeddings.weight.data = torch.from_numpy(
            symbol_injection(
                id_to_token, 0,
                model.embedder.embeddings[0].embeddings.weight.data.numpy(),
                pre_trained, oovs))
        
    else:
        pass  # No pretraining, just keep the random values.

    # Char embeddings are already random, so we don't need to update them.

    if torch.cuda.is_available() and args['cuda']:
        print('allocating on device: ', torch.cuda.current_device())
        model.cuda()
    model.train()

    optimizer = get_optimizer(model, config, state=None)
    return model, id_to_token, id_to_char, optimizer, data

def train(epoch, model, optimizer, data, choose_model):
    """
    Train for one epoch.
    """
    log_period=50
    loss_log={}
    avg_loss=0.0
    for batch_id, (qids, passages, queries, answers, _, p_labels, num_p_perQ, is_selects) in tqdm(enumerate(data)): # mapping is not used for training
        
        if choose_model.find('e2e')>=0:
            start_probs, end_probs, ranking_scores = model(passages[:2], passages[2], queries[:2], queries[2], num_p_perQ, p_labels)     
            loss = model.get_loss(start_probs, end_probs, ranking_scores, answers[:, 0], answers[:, 1], p_labels, num_p_perQ)
        elif choose_model.find('multitask')>=0:
            start_probs, end_probs, ranking_scores = model(passages[:2], passages[2], queries[:2], queries[2], num_p_perQ, p_labels)     
            loss = model.get_loss(start_probs, end_probs, ranking_scores, answers[:, 0], answers[:, 1], p_labels, num_p_perQ, is_selects)
        elif choose_model=='rar':
            start_probs, end_probs, ranking_scores = model(passages[:2], passages[2], queries[:2], queries[2])
            loss = model.get_loss(start_probs, end_probs, ranking_scores, answers[:, 0], answers[:, 1], p_labels, is_selects)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss+=float(loss.item())
        if batch_id % log_period == 0:
            loss_log[str(epoch)+'-'+str(batch_id)]= avg_loss/log_period
            print('epoch %d loss per log_period: %.6f' % (epoch, avg_loss/log_period))
            avg_loss=0.0
        
    return loss_log


def main(choose_model):
    """
    Main training program.
    """
    args = yaml.load(open('args.yaml'))
    print(args)
    if choose_model=='rar':
        config_filepath = os.path.join('config_rar.yaml')
    else:
        config_filepath = os.path.join('config.yaml')
    with open(config_filepath) as f:
        config = yaml.load(f)

    checkpoint, training_state, epoch = try_to_resume(
            args['force_restart'], args['exp_folder'], args['checkpoint'])

    if checkpoint:
        print('Resuming training...')
        model, id_to_token, id_to_char, optimizer, data = reload_state(
            checkpoint, training_state, config, args)
    else:
        print('Preparing to train...')
        model, id_to_token, id_to_char, optimizer, data = init_state(
            config, args)

        checkpoint = h5py.File(os.path.join(args['exp_folder'], args['checkpoint']))
        checkpointing.save_vocab(checkpoint, 'vocab', id_to_token)
        checkpointing.save_vocab(checkpoint, 'c_vocab', id_to_char)
        

    if torch.cuda.is_available() and args['cuda']:
        data.tensor_type = torch.cuda.LongTensor

    train_for_epochs = config.get('training', {}).get('epochs')
    if train_for_epochs is not None:
        epochs = range(epoch, train_for_epochs)# if restart ,then epoch(from try to resume)is 0
        print('num of epochs for training: ', train_for_epochs)
    else:
        epochs = itertools.count(epoch)
    loss_record={}
    for epoch in epochs:
        print('Starting epoch', epoch)
        loss_log = train(epoch, model, optimizer, data, choose_model)
        loss_record.update(loss_log)
        print('finish epoch {} save checkpoint...'.format(epoch))
        checkpointing.checkpoint(model, epoch, optimizer,
                                 checkpoint, args['exp_folder'], args['checkpoint'])
    json.dump(loss_record,open(args['loss_record'],'w'))
    
    return


if __name__ == '__main__':
    choose_model = sys.argv[1]
    if choose_model=='e2e_bidafcknrm':
        from models.e2e_bidafcknrm import MyModel
    elif choose_model=='e2e_bidafknrm':
        from models.e2e_bidafknrm import MyModel
    elif choose_model=='multitask_bidafcknrm':
        from models.multitask_bidafcknrm import MyModel
    elif choose_model=='multitask_bidafcknrm':
        from models.multitask_bidafcknrm import MyModel
    elif choose_model=='rar':
        from models.rar import RarModel as MyModel
    else:
        print('No such model yet.')
        exit()
    main(choose_model)
