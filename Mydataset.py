"""
Module used to manage data:
    Basic loading and structuring for training and testing;
    Tokenization, with vocabulary creation;
    Injection of new tokens into existing vocabulary (with random or
    pre-trained embeddings)
    Conversion into Dataset for training, etc.
"""
import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from collections import Counter
import random
from util.text_input import rich_tokenize, add_vocab
from util.squad_util import squad_tokenize


def load_data(source, debug, test=False, span_only=False, concat=False):
    """
    Load the data, and use it to create example triples.
    Input is a dictionary, output is a list of example triples, with added
    query id (query ids are not unique, all answers are included).

    :param: source: input dictionary, loaded from a json file.
    :param: span_only (bool): only keeps answers that are span in a passage.
    :param: answered_only (bool): only keeps answered questions.

    :return: a list of (qid, passage, query, (answer_start, answer_stop))
            tuples.
    :return: filtered_out: a set of qid that were filtered.
             reasons for filtered out: No Answer Present/ >=10 passages / >=1 answers 
    If there's no answer, answer_start = 0, answer_stop = 0.
    If the answer is not a span, answer_start = 0, answer_stop = len(passage)
    """

    query_ids = source['query_id']
    queries = source['query']
    passages = source['passages']
    answers = source.get('answers', {})# if no answers, return {}

    flat = ((qid, passages[qid], queries[qid], answers.get(qid))
            for qid in query_ids)
    if test:
        if concat:
            organized, filtered_out = _organize_concat_test(flat,debug)
        else:
            organized, filtered_out = _organize_test(flat, debug)
    else:
        if concat:
            organized, filtered_out = _organize_concat(flat, debug)
        else:
            organized, filtered_out = _organize(flat, span_only, debug)
    return organized, filtered_out

def _organize_concat_test(flat, debug):
    """
    Filter the queries and consolidate the answer as needed.
    """
    filtered_out = set() # qids to be filtered out
    f_forAnswer=0
    f_forM=0
    #f_forPnum=0
    organized = []# (qid, passage, query, (start, end), passage_label)
    if debug:
        only_choose = 1000
        choosed = set()
    print('organizing data...')
    for qid, passages, query, answers in tqdm(flat):
        assert type(answers)==list
        
        if debug:
            if qid not in choosed and len(choosed)>=only_choose:
                break # enough debug data has been extracted

        if answers==['No Answer Present.']:
            filtered_out.add(qid)
            f_forAnswer+=1
            continue  # Skip non-answered queries
        if len(passages)==0:
            filtered_out.add(qid)
            f_forM+=1
            continue
        concate_text = ' '.join(passage['passage_text'] for passage in passages)
        concate_passage={'attr': 'concate', 'passage_text': concate_text}
        organized.append((qid, concate_passage, query, (0, 1), 0))
        if debug and (qid not in filtered_out):# only count for not filtered out data
            choosed.add(qid)


    print('data points generated: ', len(organized))
    print('queried filtered out: ', len(filtered_out))
    print('filtered out for answer: ',f_forAnswer)
    print('filtered out for no matching', f_forM)
    return organized, filtered_out

def _organize_test(flat, debug):
    """
    Filter the queries and consolidate the answer as needed.
    """
    filtered_out = set() # qids to be filtered out
    f_forM=0
    f_forAnswer=0
    #f_forPnum=0
    organized = []# (qid, passage, query, (start, end), passage_label)
    if debug:
        only_choose = 1000
        choosed = set()
    print('organizing data...')
    for qid, passages, query, answers in tqdm(flat):
        #assert type(answers)==list
        
        if debug:
            if qid not in choosed and len(choosed)>=only_choose:
                break # enough debug data has been extracted
        
        if answers==['No Answer Present.']:
            filtered_out.add(qid)
            f_forAnswer+=1
            continue  # Skip non-answered queries
        matching=set()
        # for test(inference), get every data whatsoever
        not_matching=set()
        for ans in answers:
            if len(ans)==0:
                continue
            for ind, passage in enumerate(passages):
                if ind in matching:
                    continue
                if ind in not_matching:
                    continue
                pos = passage['passage_text'].find(ans)
                if pos>=0:
                    if passage.get('is_selected', False):# if the passage is selected but didn't find answer, (0,p_len),set plabel==1 for now
                        matching.add(ind)
                        organized.append((qid, passage, query, (pos, pos+len(ans)), 1, 1))
                    else:# not is_seleted and didn't find answer, (0,0)
                        matching.add(ind)
                        organized.append((qid, passage, query, (pos, pos+len(ans)), 1, 0))
                else:
                    if passage.get('is_selected', False):
                        not_matching.add(ind)
                        organized.append((qid, passage, query, (0,1), 0, 1))
                    else:
                        organized.append((qid, passage, query, (0,1), 0, 0))
                        not_matching.add(ind)
        if len(matching) == 0: # if no passage found in the query
            filtered_out.add(qid)
            f_forM+=1
        if debug and (qid not in filtered_out):# only count for not filtered out data
            choosed.add(qid)


    print('data points generated: ', len(organized))
    print('queried filtered out: ', len(filtered_out))
    print('filtered out for no matching', f_forM)
    print('filtered out for no answer', f_forAnswer)
    return organized, filtered_out

def _organize_concat(flat, debug):
    """
    Filter the queries and consolidate the answer as needed.
    concate all passages
    NOTE: specifically for bi-daf baseline exp
    """
    filtered_out = set() # qids to be filtered out
    f_forAnswer=0
    #f_forPnum=0
    f_forM=0
    organized = []# (qid, passage, query, (start, end), passage_label)
    if debug:
        only_choose = 1000
        choosed = set()
    print('Mode: concate')
    print('Organizing data...')
    for qid, passages, query, answers in tqdm(flat):
        assert type(answers)==list
        
        if debug:
            if qid not in choosed and len(choosed)>=only_choose:
                break # enough debug data has been extracted

        if answers is None or answers==['No Answer Present.'] or len(answers)>1:
            filtered_out.add(qid)
            f_forAnswer+=1
            continue  # Skip non-answered queries
        # concate passages
        concate_text = ' '.join(passage['passage_text'] for passage in passages)
        concate_passage={'attr': 'concate', 'passage_text': concate_text}
        with_label=0
        for ans in answers:
            if len(ans) == 0:
                continue
            pos = concate_passage['passage_text'].find(ans)
            if pos >= 0:
                organized.append((qid, concate_passage, query, (pos, pos+len(ans)),1))
                with_label+=1
        # Went through the whole thing. If there's still not match, then it got
        # filtered out.
        if with_label ==0: # no matching
            filtered_out.add(qid)
            f_forM+=1
            continue
        if debug and (qid not in filtered_out):# only count for not filtered out data
            choosed.add(qid)


    print('data points generated: ', len(organized))
    print('queried filtered out: ', len(filtered_out))
    print('filtered out for answer: ',f_forAnswer)
    print('filtered out for no matching', f_forM)
    return organized, filtered_out

def _organize(flat, span_only, debug):
    """
    Filter the queries and consolidate the answer as needed.
    """
    filtered_out = set() # qids to be filtered out
    f_forAnswer=0
    #f_forPnum=0
    f_forM=0
    organized = []# (qid, passage, query, (start, end), passage_label)
    if debug:
        only_choose = 1000
        choosed = set()
    print('organizing data...')
    for qid, passages, query, answers in tqdm(flat):
        assert type(answers)==list
        
        if debug:
            if qid not in choosed and len(choosed)>=only_choose:
                break # enough debug data has been extracted

        if answers is None or answers==['No Answer Present.'] or len(answers)>1:
            filtered_out.add(qid)
            f_forAnswer+=1
            continue  # Skip non-answered queries
        with_label=0
        matching = set()
        pairs = list()
        for ans in answers:
            if len(ans) == 0:
                continue
            for ind, passage in enumerate(passages):# first find exact matched span
                pos = passage['passage_text'].find(ans)
                if pos >= 0:
                    matching.add(ind)# passage matched
                    if passage['is_selected']:
                        pairs.append((qid, passage, query, (pos, pos+len(ans)),1,1))
                    else:
                        pairs.append((qid, passage, query, (pos, pos+len(ans)),1,0))
                    with_label+=1
        # OK, found all spans.
        if not span_only:
            for ind, passage in enumerate(passages):
                if ind in matching:
                    continue
                if passage.get('is_selected', False):# if the passage is selected but didn't find answer, (0,p_len),set plabel==1 for now
                    matching.add(ind)
                    pairs.append((qid, passage, query, (0, 1), 0, 1))
                else:# not is_seleted and didn't find answer, (0,0)
                    matching.add(ind)
                    pairs.append((qid, passage, query, (0, 1), 0, 0))
        # Went through the whole thing. If there's still not match, then it got
        # filtered out.
        if with_label ==0: # no matching
            filtered_out.add(qid)
            f_forM+=1
            continue
        organized.extend(pairs)
        if debug and (qid not in filtered_out):# only count for not filtered out data
            choosed.add(qid)


    print('data points generated: ', len(organized))
    print('queried filtered out: ', len(filtered_out))
    print('filtered out for answer: ',f_forAnswer)
    print('filtered out for no matching', f_forM)
    return organized, filtered_out

def gen_vocab(data, token_to_id, char_to_id, min_count):
    '''
    Generate vocab and cvocab
    Make token_to_id and char_to_id
    '''
    vocab_counter = Counter()
    char_counter = Counter()
    for qid, passage, query, (start, stop), label in tqdm(data):
        vocab_counter, char_counter = add_vocab(query, vocab_counter, char_counter)
        vocab_counter, char_counter = add_vocab(passage['passage_text'], vocab_counter, char_counter)
    '''
    print('total tokens: ',len(vocab_counter))
    print('total ctokens: ', len(char_counter))
    freqdic={}
    for tok, freq in vocab_counter.items():
        if freq in freqdic:
            freqdic[freq]+=1
        else:
            freqdic[freq]=1
    print('freq 1: ', freqdic[1])
    print('freq 2: ', freqdic[2])
    print('freq 3: ', freqdic[3])
    print('freq 4: ', freqdic[4])
    print('freq 5: ', freqdic[5])
    cfreqdic = {}
    for c, freq in char_counter.items():
        if freq in cfreqdic:
            cfreqdic[freq]+=1
        else:
            cfreqdic[freq]=1
    print('freq 1: ', cfreqdic[1])
    print('freq 2: ', cfreqdic[2])
    print('freq 3: ', cfreqdic[3])
    print('freq 4: ', cfreqdic[4])
    print('freq 5: ', cfreqdic[5])
    return
    '''
    for tok, freq in vocab_counter.items():
        if freq >= min_count:
            token_to_id[tok] = len(token_to_id) 
    for c, freq in char_counter.items():
        if freq >= min_count:
            char_to_id[c] = len(char_to_id) 
    return token_to_id, char_to_id



def tokenize_data(data, token_to_id, char_to_id, update=False, limit=None):
    """
    Tokenize a data set, with mapping of tokens to index in origin.
    Also create and update the vocabularies.

    :param: data: a flat, organize view of the data, as a list of qid, passage,
            query and answer indexes.
    :param: vocab: a dict of token to id; updated.
    :param: c_vocab: a dict of char to id; update.

    :return: a tokenized view of the data, as a list of qid, passage, query,
    answer indexes, and token to char indexes mapping.
    Passage and queries are tokenized into a tuple (token, chars).
    Answer indexes are start:stop range of tokens.
    """
    tokenized = []
    #print('tokenizing and making vocabs...')
    for qid, passage, query, (start, stop), label, is_select in data:
        q_tokens, q_chars, _, _, _ = \
            rich_tokenize(query, token_to_id, char_to_id, update)
        p_tokens, p_chars, _, _, mapping = \
            rich_tokenize(passage['passage_text'],
                          token_to_id, char_to_id, update)
        # locate start and end position
        if start == 0 and stop == 1:
            pass  # No answer; nop, since 0 == 0
        elif start == 0 and stop == len(passage['passage_text']):
            stop = len(p_tokens)  # Now point to just after last token.
        else:
            t_start = None
            t_end = len(p_tokens)
            for t_ind, (_start, _end) in enumerate(mapping):
                if start < _end:# start 一定大于它前面词的end
                    t_start = t_ind# locate the start word
                    break
            assert t_start is not None
            for t_ind, (_start, _end) in \
                    enumerate(mapping[t_start:], t_start):# start counting from t_start
                if stop < _start:
                    t_end = t_ind
                    break
            start = t_start  # Now point to first token in answer.
            stop = t_end  # Now point to AFTER the last token in answer.

        # Keep or not based on length of passage.
        if limit is not None and len(p_tokens) > limit:
            if stop <= limit:
                # Passage is too long, but it can be trimmed.
                p_tokens = p_tokens[:limit]
            else:
                # Passage is too long, but it cannot be trimmed.
                continue

        tokenized.append(
            (qid,
             (p_tokens, p_chars),
             (q_tokens, q_chars),
             (start, stop),
             mapping, 
             label, is_select))

    return tokenized


def symbol_injection(id_to_symb, start_at, embedding, pre_trained_source, random_source):
    """
    Inject new symbols into an embedding.
    If possible, the new embedding are retrieved from a pre-trained source.
    Otherwise, they get a new random value, using the random source.

    Will also overwrite embedding[start_at:].
    """
    if start_at == len(id_to_symb):
        return embedding  # Nothing to do.
    dim = embedding.shape[1]
    assert start_at <= len(id_to_symb)
    assert start_at <= len(embedding)
    if start_at > 0:
        embedding = embedding[:start_at]
        augment_by = len(id_to_symb) - start_at
        augment = np.empty((augment_by, dim), dtype=embedding.dtype)
        embedding = np.concatenate((embedding, augment), axis=0)
    else:
        embedding = np.empty((len(id_to_symb), dim), dtype=embedding.dtype)

    for id_ in range(start_at, len(id_to_symb)):
        symbol = id_to_symb[str(id_)]
        embedding[id_] = pre_trained_source(symbol, dim, random_source)
    return embedding

def symbol_augment(new_symbs, vocab_to_renew, start_at, embedding, pre_trained_source, random_source):
    '''
    augment new symbols into the model's embedding,
    works similarly to the above function symbol_injection
    but can add new symbols into the vocabs in the meantime
    '''
    if len(new_symbs) == 0:
        return embedding
    assert start_at <= len(embedding)
    assert start_at > 0 # or it is not augment, should use symbol injection instead
    dim = embedding.shape[1]
    embedding = embedding[:start_at]
    augment_by = len(id_to_symb) - start_at
    augment = np.empty((augment_by, dim), dtype=embedding.dtype)
    embedding = np.concatenate((embedding, augment), axis=0)

    for id_ in range(start_at, start_at+len(new_symbs)):
        symbol = new_symbs[id_]
        vocab_to_renew[symbol] = int(id_)
        embedding[id_] = pre_trained_source(symbol, dim, random_source)
    return embedding, vocab_to_renew


class SymbolEmbSource(object):
    """
    Base class for symbol embedding source.
    """

    def __init__(self):
        return

    def get_rep(self, symbol, dim):
        return None

    def __call__(self, symbol, dim, fallback=None):
        rep = self.get_rep(symbol, dim)# try to get the rep for symbol
        if rep is None and fallback is not None:# if can't get(i.e. oovs), then find the rep from random instead
            rep = fallback(symbol, dim)

        if rep is None:
            raise ValueError('Symbol [%s] cannot be found' % str(symbol))
        return rep


class SymbolEmbSourceText(SymbolEmbSource):
    """
    Load pre-trained embedding from a file object, saving only symbols of
    interest.
    If none, save all symbols.
    The saved symbols can then be retrieves

    Assumes that base_file contains line, with one symbol per line.
    """

    def __init__(self, base_file, symbols_of_interest, dtype='float32'):
        self.symbols = {}

        if symbols_of_interest is not None:
            def _skip(symbol):
                return symbol not in symbols_of_interest
        else:
            def _skip(symbol):
                return False

        dim = None
        for line in base_file:
            line = line.strip().split()
            if not line or _skip(line[0]):# line is None or token not in tokens set
                continue
            if dim is None:
                dim = len(line) - 1
            else:
                if len(line) != dim + 1:# bad line?
                    continue
            symbol = line[0]
            rep = np.array([float(v) for v in line[1:]], dtype='float32')
            self.symbols[symbol] = rep
        return

    def get_norm_stats(self, use_cov=True):
        """
        Assumes that the representation are normally distributed, and return
        the mean and standard deviation of that distribution.
        """
        data = np.array(list(self.symbols.values()))
        if use_cov:
            cov = np.cov(data, rowvar=False)
        else:
            cov = data.std(0)
        return data.mean(0), cov

    def get_rep(self, symbol, dim):
        """
        Get the representation for a symbol, and confirm that the dimensions
        match. If everything matches, return the representation, otherwise
        return None.
        """
        rep = self.symbols.get(symbol)
        if rep is not None and len(rep) != dim:
            rep = None
        return rep


class SymbolEmbSourceNorm(SymbolEmbSource):
    """
    Create random representation for symbols.
    """

    def __init__(self, mean, cov, rng, use_voc=False, cache=False):

        self.mean = mean
        self.cov = cov
        self.use_voc = use_voc
        self.rng = rng
        self.cache = {} if cache else None
        return

    def get_rep(self, symbol, dim):
        if dim != len(self.mean):
            return None
        if self.cache and symbol in self.cache:
            return self.cache[symbol]
        if self.use_voc:
            rep = self.rng.multivariate_normal(self.mean, self.cov)
        else:
            rep = self.rng.normal(self.mean, self.cov)

        if self.cache is not None:
            self.cache[symbol] = rep
        return rep


class EpochGen(object):
    """
    Generate batches over one epoch.
    """

    def __init__(self, data, vocab, cvocab, batch_size=32, shuffle=True, sample=False,
                pipeline=False, ranking_scores=None, topk=None,
                 tensor_type=torch.LongTensor):

        self.batch_size = batch_size
        print('batch size: ', self.batch_size)
        self.shuffle = shuffle
        self.sample = sample
        print('sample? ', self.sample)
        self.tensor_type = tensor_type
        if pipeline:
            print("Pipeline selecting passages %d" % topk)
            self.pipeline=True
            self.topk=topk
            self.scores = ranking_scores
        else:
            self.pipeline=False
            self.topk=None
            self.scores=None

        def group_by_qid(data, sample):
            '''
            group by data points by qid. 
            sample: each pos + all neg, for every query
            '''
            q_data={}
            for datapoint in data:
                qid=str(datapoint[0])# turn it into string
                if qid not in q_data:
                    q_data[qid]=list()
                q_data[qid].append(datapoint)
            if self.pipeline:
                # pipeline organiza: select top k data based on the ranking score
                total_pas=0
                ranked_data={}
                for qid, vals in q_data.items():
                    scores=self.scores[qid]
                    assert len(scores)== len(vals), (qid, len(scores), len(vals))
                    paired = list(zip(scores, vals))
                    paired.sort(key=lambda x:x[0], reverse = True)
                    ranked_p = [ ex[1] for ex in paired[:self.topk]]
                    ranked_data[qid] = ranked_p
                    total_pas += len(ranked_p)
                print('After ranking, have %d queries with %d passages.'% (len(ranked_data), total_pas))
                return ranked_data
            elif not sample:
                print('Not sample, num of data = num of q = ', len(q_data))
                return q_data
            else:
                sampled_data={}
                for qid, vals in q_data.items():
                    pos=[]
                    neg=[]
                    for val in vals:
                        if val[4]==1:
                            pos.append(val)
                        else:
                            neg.append(val)
                    sampled_data[qid] = random.sample(pos,1)
                    if len(neg)>4:
                        sampled_data[qid].extend(random.sample(neg,4))
                    else:
                        sampled_data[qid].extend(neg)
                    '''
                    for ind, pval in enumerate(pos):
                        sampled_data[qid+'-'+str(ind)]=[pval]
                        sampled_data[qid+'-'+str(ind)].extend(neg)
                        #if len(neg)>4:
                        #    sampled_data[qid+'-'+str(ind)].extend(random.sample(neg,4))
                        #else:
                        #    sampled_data[qid+'-'+str(ind)].extend(neg)
                    '''
                    
                        
                print('Do sample, num of data = ', len(sampled_data))
                return sampled_data

        self.data = group_by_qid(data, self.sample)
        self.token_to_id = vocab
        self.char_to_id = cvocab
        self.query_ids = [qid for qid in self.data.keys()]
        if self.shuffle:
            np.random.shuffle(self.query_ids)
        self.n_qs = len(self.query_ids)
        print('num of queries: ', self.n_qs)
        self.iter_count = 0
        return

    def process_batch_for_length(self, sequences, c_sequences):
        """
        Assemble and pad data.
        """
        assert len(sequences) == len(c_sequences)
        lengths = Variable(self.tensor_type([len(seq) for seq in sequences]))# [p1 length, p2 length, ...]
        max_length = max(len(seq) for seq in sequences) # the longest passage
        max_c_length = max(max(len(chars) for chars in seq) # the longest word
                           for seq in c_sequences)
        #print('max_length: ', max_length)
        def _padded(seq, max_length):
            _padded_seq = self.tensor_type(max_length).zero_()
            _padded_seq[:len(seq)] = self.tensor_type(seq)
            return _padded_seq
        sequences = Variable(torch.stack(
                [_padded(seq, max_length) for seq in sequences]))# padding every passage and concate(on a new dim) them together

        def _padded_char(seq, max_length, max_c_length):
            _padded = self.tensor_type(max_length, max_c_length).zero_()
            for ind, tok in enumerate(seq):# for every word( a list of char )
                _padded[ind, :len(tok)] = self.tensor_type(tok)
            return _padded

        c_sequences = Variable(torch.stack([
            _padded_char(seq, max_length, max_c_length)
            for seq in c_sequences]))
        
        return (sequences, c_sequences, lengths)

    def prepare_batch(self, qids):
        batch_data=list()
        num_p_perQ=dict()
        for qid in qids:
            batch_data.extend(self.data[qid])
            num_p_perQ[qid]=len(self.data[qid])
        return batch_data, num_p_perQ

    def __len__(self):
        return self.n_qs

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):

        if self.iter_count >= self.n_qs:
            self.iter_count = 0
            raise StopIteration()

        start_ind = self.iter_count
        batch_idx = self.query_ids[start_ind : min(start_ind + self.batch_size, self.n_qs)]

        # num_p_perQ={idx: len(self.data[idx]) for idx in batch_idx} # how many passages in each query
        batch_data, num_p_perQ=self.prepare_batch(batch_idx)
        batch_data=tokenize_data(batch_data, self.token_to_id, self.char_to_id, update=False)
        assert sum(num_p_perQ.values()) == len(batch_data)

        qids       = [ datapoint[0]    for datapoint in batch_data]
        passages   = [ datapoint[1][0] for datapoint in batch_data] # p_tokens ----------- list of p_tokens list
        c_passages = [ datapoint[1][1] for datapoint in batch_data] # p_chars
        queries    = [ datapoint[2][0] for datapoint in batch_data] # q_tokens
        c_queries  = [ datapoint[2][1] for datapoint in batch_data] # q_chars
        answers    = [ datapoint[3]    for datapoint in batch_data] # (start, stop) token index in p_tokens
        mappings   = [ datapoint[4]    for datapoint in batch_data]
        mappings   = np.array(mappings)
        p_labels   = [ datapoint[5]    for datapoint in batch_data] # label passage with answer span
        is_selects = [ datapoint[6]    for datapoint in batch_data]

        passages = self.process_batch_for_length(
                passages, c_passages)
        queries = self.process_batch_for_length(
                queries, c_queries)

        answers = Variable(self.tensor_type(answers))
        self.iter_count = self.iter_count + self.batch_size
        
        batch = (qids,
                 passages, queries,
                 answers,
                 mappings,
                 p_labels,
                 num_p_perQ,
                 is_selects)
        return batch
