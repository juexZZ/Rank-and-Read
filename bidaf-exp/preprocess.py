'''data preprocess '''
import argparse
import json
import os
from Mydataset import load_data, tokenize_data, gen_vocab, EpochGen


def prepare_data(args, mode='preprocessed'):
    '''
    prepare data for training from scratch
    '''
    token_to_id = {'<UNK>': 0}
    char_to_id = {'<PAD>': 0,'<UNK>':1}
    print('Loading original data...')
    if mode == 'debug':
    	debug = True
    else:
    	debug = False
    print('debug: ', debug)
    with open(args.data) as f_o:# f_o train_v2.1.json so on
        data, _ = load_data(json.load(f_o), test=args.test, concat=args.concat, span_only=args.span_only, debug=debug)
    if debug:
    	json.dump(data, open(args.exp_folder+'/'+mode+args.out_file, 'w'))
    	return
    if args.renew_vocab:
    	print('Generating vocabs...')
    	token_to_id, char_to_id = gen_vocab(data, token_to_id, char_to_id, min_count=args.min_count)
    	id_to_token = {id_: tok for tok, id_ in token_to_id.items()}# reverse the dict
    	id_to_char = {id_: char for char, id_ in char_to_id.items()}
    	print('len of vocab: ', len(id_to_token))
    	print('len of cvocab: ', len(id_to_char))
    	print('Dumping vocabs...')
    	json.dump(id_to_token, open(args.exp_folder+'/'+mode+'/vocab.json','w'))
    	json.dump(id_to_char, open(args.exp_folder+'/'+mode+'/cvocab.json','w'))
    
    print('Dumping data...')
    json.dump(data, open(args.exp_folder+'/'+mode+args.out_file, 'w'))
    return

if __name__=='__main__':
	argparser = argparse.ArgumentParser()
    argparser.add_argument("exp_folder", help="Experiment folder", default='../exp')
    argparser.add_argument("data", help="Training data", default='../../train_v2.1.json')
    argparser.add_argument("--mode", 
                            help="debug: only preprocess a 1000 examples; preprocessed: all",
                            choices=["debug", "preprocessed"],
                            default="preprocessed")
    argparser.add_argument("--test",
    						default=False,
    						help="Whether use it for test or train")
    argparser.add_argument("--min_count",
                            default=3,
                            help="Whether use it for test or train")
    argparser.add_argument("--concat",
    						default=False,
    						help="Whether concatenate the passages for a query")
    argparser.add_argument("--span_only",
    						default=True,
    						help="If true, only use passages containing answer spans.")
    argparser.add_argument("--renew_vocab",
                           default=True,
                           help="Preprocess to get vocab again. You only have to do this once")
    argparser.add_argument("--out_file",
    						help="Output data file name.")
    args = argparser.parse_args()
    if not os.path.exist(args.exp_folder):
        os.mkdir(args.exp_folder)
	print('data preprocessing...')
	print(args)
	prepare_data(args, args.mode)
