import itertools
import numpy as np
from nltk import sent_tokenize, word_tokenize

def add_vocab(text, vocab_counter, c_counter):
    '''
    tokenize text and add tokens, ctokens to the counter of them
    '''
    tokens = list(# use itertool.chain to get a list of word tokens
        itertools.chain.from_iterable(
            (token.replace("''", '"').replace("``", '"')
             for token in word_tokenize(sent))
            for sent in sent_tokenize(text)))
    for token in tokens:
        token_lower = token.lower().strip()
        vocab_counter[token_lower] += 1
        for c in token_lower:
            c_counter[c] += 1
    return vocab_counter, c_counter

def rich_tokenize(text, vocab, c_vocab, update):
    tokens = list(# use itertool.chain to get a list of word tokens
        itertools.chain.from_iterable(
            (token.replace("''", '"').replace("``", '"')
             for token in word_tokenize(sent))
            for sent in sent_tokenize(text)))
    #print(tokens)
    length = len(tokens)
    #print(length)
    mapping = np.zeros((length, 2), dtype='int32')
    c_lengths = np.zeros(length, dtype='int32')
    start = 0
    for ind, token in enumerate(tokens):
        _start = text.find(token, start)# find start position of each token
        t_l = len(token)
        if _start < 0 and token[0] == '"':# 因为引号问题所以没有找到词
            t_l = 2
            _a = text.find("''"+token[1:], start)
            _b = text.find("``"+token[1:], start)
            if _a != -1 and _b != -1:
                _start = min(_a, _b)
            elif _a != -1:
                _start = _a
            else:
                _start = _b
        start = _start
        assert start >= 0
        mapping[ind, 0] = start
        mapping[ind, 1] = start + t_l
        c_lengths[ind] = t_l
        start = start + t_l

    if update:# update vocab all lower() && strip()
        character_ids = [
            [c_vocab.setdefault(c, len(c_vocab)) for c in token.lower().strip()]
            for token in tokens]
        token_ids = [
            vocab.setdefault(token.lower().strip(), len(vocab)) for token in tokens]
    else:
        character_ids = [# return 1 as id if not find in the vocab
            [c_vocab.get(c, 1) for c in token.lower().strip()]
            for token in tokens]
        token_ids = [
            vocab.get(token.lower().strip(), 0) for token in tokens]

    return token_ids, character_ids, length, c_lengths, mapping.tolist()


def pad_to_size(token_ids, character_ids, t_length, c_length):
    padded_tokens = np.zeros((1, t_length), dtype='int32')
    padded_characters = np.zeros((1, t_length, c_length), dtype='int32')
    padded_tokens[0, :len(token_ids)] = token_ids
    for ind, _chars in enumerate(character_ids):
        padded_characters[0, ind, :len(_chars)] = _chars
    return padded_tokens, padded_characters


def text_as_batch(text, vocab, c_vocab):
    tokens, chars, length, c_lengths, mapping = \
        rich_tokenize(text, vocab, c_vocab, update=False)
    tokens, chars = pad_to_size(tokens, chars, length, max(5, c_lengths.max()))
    length = np.array([length])
    return tokens, chars, length, mapping
