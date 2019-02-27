'''bidaf + cknrm multitask '''

import torch
from torch import nn
from torch.nn.functional import nll_loss, binary_cross_entropy
from torch.autograd import Variable
import numpy as np
import h5py
from models.embeddings import *
from models.highway import Highways
from models.MyRanker import cknrm
#from models.MyRanker import knrm

class MyModel(nn.Module):
    """
    Bidirectional attention flow model for question answering.
    """

    def __init__(self,
                 embedder,
                 ranker_embedder,
                 ranker_embedding_dim,
                 num_bins,
                 num_highways,
                 num_lstm,
                 hidden_size,
                 dropout,
                 ifcuda):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.bidir_hidden_size = 2*hidden_size
        self.embedder = embedder
        self.ranker_embedder = ranker_embedder
        self.highways = Highways(embedder.output_dim, num_highways)
        self.seq_encoder = nn.LSTM(embedder.output_dim,
                                   hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   dropout=0,
                                   bidirectional=True)
        self.extractor = nn.LSTM(4*self.bidir_hidden_size,
                                 hidden_size,
                                 num_layers=num_lstm,
                                 batch_first=True,
                                 dropout=0,
                                 bidirectional=True)
        self.end_encoder = nn.LSTM(7*self.bidir_hidden_size,
                                   hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   dropout=0,
                                   bidirectional=True)
        self.attention = AttentionMatrix(self.bidir_hidden_size)

        self.ranker = cknrm(num_bins, ranker_embedding_dim, ifcuda)
        # Second hidden_size is for extractor.
        self.start_projection = nn.Linear(
            4*self.bidir_hidden_size + self.bidir_hidden_size, 1)
        self.end_projection = nn.Linear(
            4*self.bidir_hidden_size + self.bidir_hidden_size, 1)

        if dropout and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda nop: nop
        return

    @classmethod
    def _parse_config(cls, config, vocab, c_vocab):
        num_tokens = len(vocab)
        num_chars = len(c_vocab)

        token_embs = TokenEmbedding(
            num_tokens, config['embedding_dim'],
            output_dim=None, static = False)

        _config = config['characters']
        char_embs = CharEmbedding(
            num_chars,
            _config.get('dim', 16),
            _config.get('num_filters', 100),
            _config.get('filter_sizes', [5]))
        ranker_config = config['cknrm']
        args = (
                CatEmbedding([token_embs, char_embs]),
                #UpdatedTokenEmbedding(num_tokens,ranker_config['embedding_dim']),
                None,
                ranker_config.get('embedding_dim', 300),
                ranker_config.get('n_bins', 21),
                config.get('num_highways', 2),
                config.get('num_lstm', 2),
                config.get('hidden_size', 100),
                config.get('dropout', 0.2))
        return args

    @classmethod
    def from_config(cls, config, vocab, c_vocab, ifcuda):
        """
        Create a model using the model description in the configuration file.
        """
        model = cls(*cls._parse_config(config, vocab, c_vocab), ifcuda)
        return model


    @classmethod
    def _pack_and_unpack_lstm(cls, input, lengths, seq_encoder):
        """
        LSTM, when using batches, should be called with a PackedSequence.
        Doing this will deal with the different lengths in the batch.
        PackedSequence must be created from batches where the sequences are
        stored with decreasing lengths.

        _pack_and_unpack_lstm handles this issue.
        It re-orders its input, pack it, sends it through the LSTM and finally
        restore the original order.

        This is not general purpose: in particular, it does not handle initial
        and final states.
        """
        s_lengths, indexes = lengths.sort(0, descending=True)# sorted by length
        s_input = input.index_select(0, indexes)# re ordered input

        i_range = torch.arange(lengths.size()[0]).type_as(lengths.data)# batch size
        i_range = Variable(i_range)
        _, reverses = indexes.sort(0, descending=False)# tell how to recover index
        reverses = i_range.index_select(0, reverses)

        packed = nn.utils.rnn.pack_padded_sequence(
            s_input, s_lengths.data.tolist(), batch_first=True)

        output, _ = seq_encoder(packed)
        # Unpack and apply reverse index.
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)
        output = output.index_select(0, reverses)

        return output

    @classmethod
    def _apply_attention_to_question(cls, similarity, enc_question, mask):
        """
        Apply attention to question, while masking for lengths
        """
        # similarity: [batch, m_p, m_q]
        # enc_question: [batch, m_q, hidden_size]
        # mask: [batch, m_q]
        batch, m_p, m_q = similarity.size()

        _sim = similarity.view(
            batch*m_p, m_q)
        # mask (batchsize, squence)
        tmp_mask = mask.unsqueeze(1).expand(
            batch, m_p, m_q).contiguous().float()
        tmp_mask = tmp_mask.view(batch*m_p, m_q)
        _sim = nn.functional.softmax(_sim*tmp_mask - (1-tmp_mask)*1e20, dim=1)
        _sim = _sim.view(batch, m_p, m_q)

        out = _sim.bmm(enc_question) # [batchsize, m_p, embed]
        return out

    @classmethod
    def _apply_attention_to_passage(cls, similarity, enc_passage, p_mask, q_mask):
        """
        Apply attention to passage, while masking for lengths.
        """
        # similarity: [batch, m_p, m_q]
        # enc_passage: [batch, m_p, hidden_size]
        # p_mask: [batch, m_p]
        # q_mask: [batch, m_q]
        batch, m_p, m_q = similarity.size()

        # Mask the similarity
        tmp_mask = q_mask.unsqueeze(1).expand(
            batch, m_p, m_q).contiguous().float()
        similarity = similarity * tmp_mask - (1-tmp_mask)*1e20
        # Pick the token in the question with the highest similarity with a
        # given token in the passage as the similarity between the entire
        # question and that passage token
        similarity = similarity.max(dim=2)[0]
        # Final similarity: [batch, m_p]

        tmp_mask = (1-p_mask)
        tmp_mask = 1e20*tmp_mask
        similarity = nn.functional.softmax(similarity*p_mask - tmp_mask, dim=1)
        out = similarity.unsqueeze(1).bmm(enc_passage).squeeze(1)
        return out

    def _encode(self, features, lengths):
        """
        Encode text with the embedder, highway layers and initial LSTM.
        """
        embedded, _ = self.embedder(features)
        # embedded = self.embedder(features)
        batch_size, num_tokens = embedded.size()[:2]
        embedded = self.highways(embedded.view(
            batch_size*num_tokens, -1))
        embedded = embedded.view(batch_size, num_tokens, -1)
        encoded = self.dropout(self._pack_and_unpack_lstm(
            embedded, lengths, self.seq_encoder))
        return encoded

    @classmethod
    def _create_mask_like(cls, lengths, like):
        """
        Create masks based on lengths. The mask is then converted to match the
        type of `like`, a Variable.
        """
        mask = torch.zeros(like.size()[:2])
        for ind, _length in enumerate(lengths.data):
            mask[ind, :_length] = 1
        mask = mask.type_as(like.data)
        mask = Variable(mask, requires_grad=False)
        return mask

    def _attention(self, enc_passage, enc_question, p_mask, q_mask):
        """
        Get and apply the attention matrix for the passage and question.
        """
        batch_size, p_num_tokens = enc_passage.size()[:2]
        # Similarity score (unnormalized) between passage and question.
        # Shape: [batch, p_num_tokens, q_num_tokens]
        similarity = self.attention(enc_passage, enc_question)

        # Shape: [batch, p_num_tokens, hidden_size]
        question_in_passage = self._apply_attention_to_question(
            similarity, enc_question, q_mask)

        # Shape: [batch, hidden_size]
        passage_in_question = self._apply_attention_to_passage(
            similarity, enc_passage, p_mask, q_mask)
        passage_in_question = passage_in_question.unsqueeze(1).expand(
            batch_size, p_num_tokens, self.bidir_hidden_size)
        return question_in_passage, passage_in_question

    def _get_ranking_score(self, passage, p_lengths, question, q_lengths, num_p_perQ):
        '''
        get ranking scores for each qp pairs (not normalized)
        passage[0], question[0] are the tokens indices, currently exclude char embeddinngs
        '''
        mask_p = self._create_mask_like(p_lengths, passage[0])
        mask_q = self._create_mask_like(q_lengths, question[0])
        _, p_t_embed = self.embedder(passage)
        _, q_t_embed = self.embedder(question)
        #p_t_embed = self.ranker_embedder(passage[0])
        #q_t_embed = self.ranker_embedder(question[0])
        assert p_t_embed.size()[0] == passage[0].size()[0]
        assert p_t_embed.size()[1] == passage[0].size()[1]
        assert q_t_embed.size()[0] == question[0].size()[0]
        assert q_t_embed.size()[1] == question[0].size()[1]
        ranking_scores = self.ranker(q_t_embed, p_t_embed, mask_q, mask_p)
        #ranking_scores2 = self.ranker2(q_t_embed, p_t_embed, mask_q, mask_p)
        #print (ranking_scores)
        #print ("huahua")
        #print (ranking_scores2)
        #print (ranking_scores.size()) 
        # check if size of ranking_scores == [batchsize]
        assert len(ranking_scores) == p_lengths.size()[0]

        # normalize ranking scores with respect to each query
        start=0
        log_result=[]# for the MRC task
        result=[]# for the IR task (multitask learning)
        assert len(ranking_scores) == sum(num_p_perQ.values())
        for num in list(num_p_perQ.values()):
            perQ = ranking_scores[start:start+num] # get each query
            #perQ2 = ranking_scores2[start:start+num]
            perQ = perQ - perQ.max()
            
            #log_norm_perQ = nn.functional.log_softmax(perQ2, dim=0)
            #log_norm_perQ = nn.functional.softmax(perQ, dim=0)
            norm_perQ = nn.functional.softmax(perQ,dim=0).clamp(min=0,max=1)
            #log_result.append(log_norm_perQ)
            result.append(norm_perQ)
            start = start + num
        #log_norm_ranking_scores = torch.cat(tuple(log_result), dim=0)
        norm_ranking_scores = torch.cat(tuple(result),dim=0)

        #assert len(norm_ranking_scores) == len(ranking_scores)

        return norm_ranking_scores#, log_norm_ranking_scores

    def _filter_for_reader(self, labels):
        if self.training == False:
            return None
        else:
            '''
            select those label=1(contain answer) passages, and only send them into reader
            '''
            labels = Variable(torch.ByteTensor(labels).cuda(), requires_grad=False)
            index = labels.nonzero().squeeze()
            return index


    def forward(self, passage, p_lengths, question, q_lengths, num_p_perQ, p_labels):
        """
        Forward pass
        """

        # Ranker
        #combin_embedded, token_embedded = self.embedder(features)
        # Get Ranking Scores for each passage
        ranking_scores = self._get_ranking_score(passage, p_lengths, question, q_lengths, num_p_perQ)
        #print (ranking_scores)
        # only pos passages go to reader
        index = self._filter_for_reader(p_labels)
        if index is not None:
            #print('reader only handle true')
            passage = (passage[0].index_select(0,index), passage[1].index_select(0,index))
            question = (question[0].index_select(0,index), question[1].index_select(0,index))
            p_lengths = p_lengths.index_select(0,index)
            q_lengths = q_lengths.index_select(0,index)
        

        # Encode the text
        enc_passage = self._encode(passage, p_lengths)
        enc_question = self._encode(question, q_lengths)

        # Get the sizes
        batch_size, p_num_tokens = enc_passage.size()[:2] #(batch size, num token, embed dim)
        q_batch_size, q_num_tokens = enc_question.size()[:2]
        assert batch_size == q_batch_size
        assert batch_size == p_lengths.size()[0]# dim 0 == batch size
        assert batch_size == q_lengths.size()[0]

        # Create the masks
        p_mask = self._create_mask_like(p_lengths, enc_passage)
        q_mask = self._create_mask_like(q_lengths, enc_question)

        # Get similarities and apply the attention mechanism
        # [batch, p_num_tokens, hidden_size]
        (question_in_passage, passage_in_question) = \
            self._attention(enc_passage, enc_question, p_mask, q_mask)

        # Concatenate the passage and similarities, then use a LSTM stack to
        # extract features.
        # 4 [b, p_num_tokens, hidden_size]
        # -> [b, n, 4*hidden_size]
        merged_passage = torch.cat([
            enc_passage,
            question_in_passage,
            enc_passage * question_in_passage,
            enc_passage * passage_in_question],
            dim=2)
        # fuse attention information
        extracted = self.dropout(self._pack_and_unpack_lstm(
            merged_passage, p_lengths, self.extractor))

        # Answer Boundary Pediction
        # Use the features to get the start point probability vectors.
        # Also use it to as attention over the features.
        start_input = self.dropout(
            torch.cat([merged_passage, extracted], dim=2))
        # [b, p_num_tokens, 4*h] -> [b, n, 1] -> [b, n]
        start_projection = self.start_projection(start_input).squeeze(2)
        # Mask
        start_logits = start_projection*p_mask + (p_mask-1)*1e20
        # And turns into probabilities
        start_probs = nn.functional.softmax(start_logits, dim=1)
        # And then into representation, as attention.
        # [b, 1, hidden_size] -> [b, p_num_tokens, hidden_size]
        start_reps = start_probs.unsqueeze(1).bmm(extracted)
        start_reps = start_reps.expand(
            batch_size, p_num_tokens, self.bidir_hidden_size)

        # Uses various level of features to create the end point probability
        # vectors.
        # [b, n, 7*hidden_size]
        end_reps = torch.cat([
            merged_passage,
            extracted,
            start_reps,
            extracted * start_reps],
            dim=2)
        enc_end = self.dropout(self._pack_and_unpack_lstm(
            end_reps, p_lengths, self.end_encoder))
        end_input = self.dropout(torch.cat([
            merged_passage, enc_end], dim=2))
        # [b, p_num_tokens, 7*h] -> [b, n, 1] -> [b, n]
        end_projection = self.end_projection(end_input).squeeze(2)
        # Mask
        end_logits = end_projection*p_mask + (p_mask-1)*1e20

        # Applies the final log-softmax to get the actual log-probability
        # vectors.
        log_start_probs = nn.functional.log_softmax(start_logits, dim=1)
        log_end_probs = nn.functional.log_softmax(end_logits, dim=1)

        return log_start_probs, log_end_probs, ranking_scores#, ranking_scores_log


    @classmethod
    def get_loss(cls, log_start_probs, log_end_probs, ranking_scores, starts, ends, p_labels, num_p_perQ, is_selects):
        """
        Get the loss, 
        """
        #p_labels = Variable(torch.ByteTensor(p_labels).cuda(), requires_grad=False)
        # p_labels = Variable(torch.FloatTensor(p_labels).cuda(), requires_grad=False)
        is_selects = Variable(torch.FloatTensor(is_selects).cuda(), requires_grad=False)
        #print(is_selects.size())
        #ranking_scores = Variable(torch.FloatTensor(ranking_scores).cuda(), requires_grad=True)
        ####################### MULTITASK RANING:: IR LOSS ###########################
        assert len(ranking_scores) == len(is_selects)
        assert min(ranking_scores)>=0 and max(ranking_scores)<=1, (ranking_scores)
        # assert min(is_selects)>=0 and max(is_selects)<=1, (is_selects)
        # assert (delog_ranking_scores>=0. & delog_ranking_scores<=1.).all()
        # assert (is_selects>=0. & is_selects<=1.).all()
        #print (ranking_scores.size())
        #print (is_selects.size())
        #print (ranking_scores)
        #print (is_selects)
        ir_loss = binary_cross_entropy(ranking_scores, is_selects)
        
        #print(log_start_probs.size())
        p_labels = Variable(torch.ByteTensor(p_labels).cuda(), requires_grad=False)
        #print(p_labels.size(), mask.size())
        starts = starts.index_select(0, p_labels.nonzero().squeeze())
        ends = ends.index_select(0, p_labels.nonzero().squeeze())
        #selected_start_probs = log_start_probs * mask
        #selected_end_probs = log_end_probs * mask
        loss = nll_loss(log_start_probs, starts) + \
                nll_loss(log_end_probs, ends-1) + \
                ir_loss
        
        return loss

    @classmethod
    def get_best_span(cls, start_log_probs, end_log_probs):
        """
        Get the best span for each passage.
        """
 
        batch_size, num_tokens = start_log_probs.size()
        # expand ranking score and then add to each token's start log probs
        # assert len(ranking_scores) == batch_size
        # ranking_scores=ranking_scores.unsqueeze(1).expand(batch_size, num_tokens)
        # start_log_probs = start_log_probs + ranking_scores

        start_end = torch.zeros(batch_size, 2).long()
        max_val = start_log_probs[:, 0] + end_log_probs[:, 0]
        max_start = start_log_probs[:, 0]
        arg_max_start = torch.zeros(batch_size).long()

        # answer span prediction
        for batch in range(batch_size):# for every passage
            _start_lp = start_log_probs[batch]
            _end_lp = end_log_probs[batch]
            for t_s in range(1, num_tokens):
                if max_start[batch] < _start_lp[t_s]:
                    arg_max_start[batch] = t_s
                    max_start[batch] = _start_lp[t_s]

                cur_score = max_start[batch] + _end_lp[t_s]
                if max_val[batch] < cur_score:
                    start_end[batch, 0] = arg_max_start[batch]
                    start_end[batch, 1] = t_s
                    max_val[batch] = cur_score

        # Place the end point one time step after the end, so that
        # passage[s:e] works.
        start_end[:, 1] += 1
        return start_end, max_val

    @classmethod
    def passage_selection(cls, qids, predictions, passages, mappings, num_p_perQ, ranking_scores=None, p_labels=None, max_val=None):
        '''
        use ranking score to get the predicted span from the scored highest passage as the final candidate span for thie query
        OR use p labels to do the same thing
        OR simply use the max P(start)*P(end)
        '''
        prepare=dict()
        candidate=dict()
        if ranking_scores is not None:
            #print('Select by ranking scores')
            if max_val is not None:
                print('choose by max_val + ranking score')
                for qid, mapping, tokens, pred, score, val in zip(qids, mappings, passages, predictions, ranking_scores, max_val):
                    if qid not in prepare:
                        prepare[qid]=list()
                    prepare[qid].append((tokens[pred[0]:pred[1]],# tokens (id)
                                    mapping[pred[0]][0], # text level start of start position
                                    mapping[pred[1]-1][1],
                                    score+val))# judge: scores, labels...
            else:
                for qid, mapping, tokens, pred, score in zip(qids, mappings, passages, predictions, ranking_scores):
                    if qid not in prepare:
                        prepare[qid]=list()
                    prepare[qid].append((tokens[pred[0]:pred[1]],# tokens (id)
                                    mapping[pred[0]][0], # text level start of start position
                                    mapping[pred[1]-1][1],
                                    score))# judge: scores, labels...
            #sort and select
            for qid, pairs in prepare.items():
                assert type(pairs) == list
                pairs.sort(key=lambda x: x[3], reverse = True)
                assert type(pairs[0]) == tuple
                candidate[qid] = pairs[0]
            return candidate

        elif p_labels is not None:
            print('Select by passage labels')
            for qid, mapping, tokens, pred, label, prob in zip(qids, mappings, passages, predictions, p_labels, max_val):
                if label == 1:
                    if qid not in prepare:
                        prepare[qid]=list()
                    prepare[qid].append((tokens[pred[0]:pred[1]],# tokens (id)
                                mapping[pred[0]][0], # text level start of start position
                                mapping[pred[1]-1][1],
                                prob))# judge: scores, labels...
            # sort and select
            for qid, pairs in prepare.items():
                assert type(pairs) == list
                pairs.sort(key=lambda x: x[3], reverse = True)
                assert type(pairs[0]) == tuple
                candidate[qid]=pairs[0]
            return candidate

        elif max_val is not None:
            # print('Select by highest prob')
            for qid, mapping, tokens, pred, judge in zip(qids, mappings, passages, predictions, max_val):
                if qid not in prepare:
                    prepare[qid]=list()
                prepare[qid].append((tokens[pred[0]:pred[1]],# tokens (id)
                                mapping[pred[0]][0], # text level start of start position
                                mapping[pred[1]-1][1],
                                judge))# judge: scores, labels...
            # sort and select
            for qid, pairs in prepare.items():
                assert type(pairs) == list
                pairs.sort(key=lambda x: x[3], reverse = True)
                assert type(pairs[0]) == tuple
                candidate[qid]=pairs[0]
            return candidate

        else:
            print('No select???')
            return





    @classmethod
    def get_combined_logits(cls, start_log_probs, end_log_probs):
        """
        Combines the start and end log probability vectors into a matrix.
        The rows correspond to start points, the columns to end points.
        So, the value at m[s,e] is the log probability of the span from s to e.
        """
        batch_size, p_num_tokens = start_log_probs.size()

        t_starts = start_log_probs.unsqueeze(2).expand(
            batch_size, p_num_tokens, p_num_tokens)
        t_ends = end_log_probs.unsqueeze(1).expand(
            batch_size, p_num_tokens, p_num_tokens)
        return t_starts + t_ends

    @classmethod
    def from_checkpoint(cls, config, checkpoint, ifcuda):
        """
        Load a model, on CPU and eval mode.

        Parameters:
            :param: config: a dictionary with the model's configuration
            :param: checkpoint: a h5 files containing the model's parameters.

        Returns:
            :return: the model, on the cpu and in evaluation mode.

        Example:
            ```
            with open('config.yaml') as f_o:
                config = yaml.load(f_o)

            with closing(h5py.File('checkpoint.h5', mode='r')) as checkpoint:
                model, vocab, c_vocab = MyModel.from_checkpoint(
                    config, checkpoint)
            model.cuda()
            ```
        """
        #vocabchkpt = h5py.File('../exp/checkpoint')
        #model_vocab = vocabchkpt['vocab']
        #model_c_vocab = vocabchkpt['c_vocab']
        model_vocab = checkpoint['vocab']
        model_c_vocab = checkpoint['c_vocab']

        model_vocab = {id_: tok for id_, tok in enumerate(model_vocab)}
        model_c_vocab = {id_: tok for id_, tok in enumerate(model_c_vocab)}

        model = cls.from_config(
                config,
                model_vocab,
                model_c_vocab,
                ifcuda)

        model.load_state_dict({
            name: torch.from_numpy(np.array(val))
            for name, val in
            checkpoint['model'].items()})
        model.eval()
        return model, model_vocab, model_c_vocab


class AttentionMatrix(nn.Module):
    """
    Attention Matrix (unnormalized)
    """

    def __init__(self, hidden_size):
        """
        Create a module for attention matrices. The input is a pair of
        matrices, the output is a matrix containing similarity scores between
        pairs of element in the matrices.

        Similarity between two vectors `a` and `b` is SIMPLIFIED, masured by
        the dot prodoct $a^Tb$ between the representation vec $a$ and $b$ in
        the tow matrices (two batches of matrices)

        Parameters:
            :param: hidden_size (int): The size of the vectors

        Variables/sub-modules:
            projection: The linear projection $W$, $C$.

        Inputs:
            :param: mat_0 ([batch, n, hidden_size] Tensor): the first matrices
            :param: mat_1 ([batch, m, hidden_size] Tensor): the second matrices

        Returns:
            :return: similarity (batch, n, m) Tensor: the similarity matrices,
            so that similarity[:, n, m] = f(mat_0[:, n], mat_1[:, m])
        """
        super(AttentionMatrix, self).__init__()
        self.hidden_size = hidden_size
        #self.projection = nn.Linear(3*hidden_size, 1)
        return

    def forward(self, mat_0, mat_1):
        """
        Forward pass.
        """
        batch, n_0, _ = mat_0.size()
        _, n_1, _ = mat_1.size()
        mat_0, mat_1 = self.tile_to_match(mat_0, mat_1)
        mat_p = mat_0*mat_1 # element-wise multiplication
        # use dot product as similarity:
        sim_pro= torch.sum(mat_p,dim=3)
        simb_size,sim_x,sim_y=sim_pro.size()
        assert simb_size == batch
        assert sim_x == n_0
        assert sim_y == n_1
        return sim_pro
        '''
        # use linear similarity
        combined = torch.cat((mat_0, mat_1, mat_p), dim=3)
        # projected down to [b, n, m]
        projected = self.projection(
            combined.view(batch*n_0*n_1, 3*self.hidden_size))
        projected = projected.view(batch, n_0, n_1)
        return projected
        '''

    @classmethod
    def tile_to_match(cls, mat_0, mat_1):
        """
        Enables broadcasting between mat_0 and mat_1.
        Both are tiled to 4 dimensions, from 3.

        Shape:
            mat_0: [b, n, e], and
            mat_1: [b, m, e].

        Then, they get reshaped and expanded:
            mat_0: [b, n, e] -> [b, n, 1, e] -> [b, n, m, e]
            mat_1: [b, m, e] -> [b, 1, m, e] -> [b, n, m, e]
        """
        batch, n_0, size = mat_0.size()
        batch_1, n_1, size_1 = mat_1.size()
        assert batch == batch_1
        assert size_1 == size
        mat_0 = mat_0.unsqueeze(2).expand(
            batch, n_0, n_1, size)
        mat_1 = mat_1.unsqueeze(1).expand(
            batch, n_0, n_1, size)
        return mat_0, mat_1


