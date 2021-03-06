# predict
#!python3
"""
Training script: load a config file, create a new model using it,
then train that model.
"""
import json
import yaml
import os.path
from tqdm import tqdm
import numpy as np
import torch
import h5py

from Mydataset import load_data, tokenize_data, gen_vocab, EpochGen
from Mydataset import SymbolEmbSourceNorm
from Mydataset import SymbolEmbSourceText
from Mydataset import symbol_augment


def try_to_resume(exp_folder,checkpointfile):
    print('Resume checkpointfile: ', exp_folder+'/'+checkpointfile)
    if os.path.isfile(exp_folder + '/'+checkpointfile):
        checkpoint = h5py.File(exp_folder + '/'+checkpointfile)
    else:
        checkpoint = None
    return checkpoint

def reload_state(checkpoint, config, args):
    """
    Reload state before predicting.
    """
    print('Loading Model...')
    token_to_id = {'<UNK>': 0}
    char_to_id = {'<PAD>': 0,'<UNK>':1}
    data=json.load(open(args['exp_folder']+args['data'], 'r'))
    model, id_to_token, id_to_char = MyModel.from_checkpoint(config['bidaf'], checkpoint, args['cuda'])

    token_to_id = {tok: id_ for id_, tok in id_to_token.items()}
    char_to_id = {char: id_ for id_, char in id_to_char.items()}

    len_tok_voc = len(id_to_token)
    len_char_voc = len(id_to_char)
    print('len of token vocabulary: ', len_tok_voc)
    print('len of char vocabulary: ', len_char_voc)
    
    if args['update_vocab']:
        print('Update vocabs...')
        new_token_id, new_char_id = gen_vocab(data, token_to_id, char_to_id, min_count=5)

        token_need = set(tok for tok, id_ in new_token_id.items() if tok not in token_to_id)
        token_need = list(token_need)
        if len(token_need) != 0:
            print('Augmenting token embeddings...')
            if args['word_rep']: # use pretrained data (eg. glove) to augment embedding
                with open(args['word_rep']) as f_o:
                    pre_trained = SymbolEmbSourceText(
                        f_o, token_need)
            else: # random augment embedding
                pre_trained = SymbolEmbSourceText([], token_need)

            cur = model.embedder.embeddings[0].embeddings.weight.data.numpy()# get the token embeddings of the checkpoint model
            mean = cur.mean(0)
            if args['use_covariance']:
                cov = np.cov(cur, rowvar=False)
            else:
                cov = cur.std(0)

            rng = np.random.RandomState(2)
            oovs = SymbolEmbSourceNorm(mean, cov, rng, args['use_covariance'])

            if args['word_rep']:
                print('Augmenting with pre-trained embeddings...')
            else:
                print('Augmenting with random embeddings...')
            augment, token_to_id = symbol_augment(
                    token_need, token_to_id, len_tok_voc,
                    model.embedder.embeddings[0].embeddings.weight.data.numpy(),
                    pre_trained, oovs)

            model.embedder.embeddings[0].embeddings.weight.data = torch.from_numpy(augment)    
    
        char_need = set(char for char, id_ in new_char_id.items() if char not in char_to_id)
        char_need = list(char_need)
        if len(char_need) != 0:
            print('Augmenting with random char embeddings...')
            pre_trained = SymbolEmbSourceText([], None)
            cur = model.embedder.embeddings[1].embeddings.weight.data.numpy()
            mean = cur.mean(0)
            if args['use_covariance']:
                cov = np.cov(cur, rowvar=False)
            else:
                cov = cur.std(0)

            rng = np.random.RandomState(2)
            oovs = SymbolEmbSourceNorm(mean, cov, rng, args['use_covariance'])

            augment, char_to_id = symbol_augment(
                    char_need, char_to_id, len_char_voc,
                    model.embedder.embeddings[1].embeddings.weight.data.numpy(),
                    pre_trained, oovs)
            model.embedder.embeddings[1].embeddings.weight.data = torch.from_numpy()
        id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
        id_to_char = {id_: char for char, id_ in char_to_id.items()}

    else:
        print('Do not update vocabs...')
    # currently not expand vocab
    data = get_loader(data, token_to_id, char_to_id, config)
    if torch.cuda.is_available() and args['cuda']:
        model.cuda()
    model.eval()

    return model, id_to_token, id_to_char, data


def get_loader(data, vocab, cvocab, config):
    data = EpochGen(
        data, vocab, cvocab,
        batch_size=config.get('predict', {}).get('batch_size', 32),
        shuffle=False, sample=False)
    return data

def get_ranker_score(model, data, args, fout):
    """ get and dumo ranking scores """
    rank_scores=dict()

    for batch_id, (qids, passages, queries, _, mappings, p_labels, num_p_perQ, is_selects) in tqdm(enumerate(data)):
        _, _, ranking_scores = model(passages[:2], passages[2], queries[:2], queries[2], num_p_perQ, p_labels)
        ranking_scores = ranking_scores.data.cpu().numpy()
        for qid, score in zip(qids, ranking_scores):
            if qid not in rank_scores:
                rank_scores[qid]=list()
            rank_scores[qid].append(score.item())
    print("Total number of queries: %d." % len(rank_scores))
    json.dump(rank_scores, fout)



def predict(model, data, args, id_to_token, fout, choose_model):
    """ Predict """
    if args['store_rank']:
        rank_scores=dict()
    for batch_id, (qids, passages, queries, _, mappings, p_labels, num_p_perQ, _) in tqdm(enumerate(data)):
        # print('{}/{}'.format(batch_id, len(data)))
        if choose_model=='rar':
            start_log_probs, end_log_probs, ranking_scores = model(passages[:2], passages[2], queries[:2], queries[2])
        else:
            start_log_probs, end_log_probs, ranking_scores = model(passages[:2], passages[2], queries[:2], queries[2], num_p_perQ, p_labels)
        
        predictions, max_val = model.get_best_span(start_log_probs.data.cpu(), end_log_probs.data.cpu())
        # size predictions [batch,start,end]
        ranking_scores = ranking_scores.data.cpu().numpy()
        if args['store_rank']:
            for qid, score in zip(qids, ranking_scores):
                if qid not in rank_scores:
                    rank_scores[qid]=list()
                rank_scores[qid].append(score.item())
        max_val = max_val.numpy()
        passages = passages[0].cpu().numpy()# only use the token part
        candidates = model.passage_selection(qids, predictions, passages, mappings,
                                            num_p_perQ, ranking_scores=ranking_scores, max_val=None)
        #candidates: {qid: best candidate answer}
        for qid, vals in candidates.items():
            toks = vals[0]
            answer = [' '.join(id_to_token[tok] for tok in toks)]
            json.dump({'query_id':int(qid), 'answers': answer}, fout)
            fout.write('\n')

    if args['store_rank']:
        print("Total number of queries: %d." % len(rank_scores))
        json.dump(rank_scores, open(args['rank_file'], 'w'))
    return


def main(choose_model):
    """
    Main prediction program.
    """
    args = yaml.load(open('predict.yaml'))
    print(args)

    config_filepath = os.path.join('config.yaml')
    with open(config_filepath) as f:
        config = yaml.load(f)

    checkpoint = try_to_resume(args['chkpt_folder'], args['checkpoint'])

    if checkpoint:
        model, id_to_token, id_to_char, data = reload_state(
            checkpoint, config, args)
    else:
        print('Need a valid checkpoint to predict.')
        return

    if torch.cuda.is_available() and args['cuda']:
        data.tensor_type = torch.cuda.LongTensor

    with open(args['dest'], 'w') as f_o:
        predict(model, data, args, id_to_token, f_o, choose_model)
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
