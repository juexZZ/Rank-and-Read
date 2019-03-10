class BertForMultiTask(PreTrainedBertModel):
    def __init__(self, config):
        super(BertForMultiTask, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.ranker = cknrm(21, config.hidden_size, True)
        self.m = nn.Sigmoid()
        self.apply(self.init_bert_weights)

    def _create_mask_like(self, lengths, like):
        mask = torch.zeros(like.size()[:2])
        for ind, _length in enumerate(lengths.data):
            mask[ind, :_length] = 1
        mask = mask.type_as(like.data)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        return mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, q_len=None, d_len=None, start_positions=None, end_positions=None, is_selects=None, has_answers=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        
        q_t_embed = torch.nn.utils.rnn.pad_sequence([sequence_output[i, 1:1+q_len[i], :] for i in range(len(q_len))], batch_first=True)
        p_t_embed = torch.nn.utils.rnn.pad_sequence([sequence_output[i, 1+q_len[i]+1:1+q_len[i]+1+d_len[i], :] for i in range(len(d_len))], batch_first=True)
        mask_p = self._create_mask_like(d_len, p_t_embed)
        mask_q = self._create_mask_like(q_len, q_t_embed)
        
        scores = self.ranker(q_t_embed, p_t_embed, mask_q, mask_p)
        scores = self.m(scores)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # ignore negative RC loss
            mask = has_answers.unsqueeze(1).repeat(1, start_logits.size(1))
            start_logits = start_logits * mask
            end_logits = end_logits * mask

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss_fct1 = BCELoss()
            ir_loss = loss_fct1(scores, is_selects)
            
            total_loss = start_loss + end_loss + ir_loss
            return total_loss
        else:
            return start_logits, end_logits, scores

class BertForEnd2End(PreTrainedBertModel):
    def __init__(self, config):
        super(BertForEnd2End, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.ranker = cknrm(21, config.hidden_size, True)
        self.apply(self.init_bert_weights)

    def _create_mask_like(self, lengths, like):
        mask = torch.zeros(like.size()[:2])
        for ind, _length in enumerate(lengths.data):
            mask[ind, :_length] = 1
        mask = mask.type_as(like.data)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        return mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, q_len=None, d_len=None, start_positions=None, end_positions=None, is_selects=None, has_answers=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        q_t_embed = torch.nn.utils.rnn.pad_sequence([sequence_output[i, 1:1+q_len[i], :] for i in range(len(q_len))], batch_first=True)
        p_t_embed = torch.nn.utils.rnn.pad_sequence([sequence_output[i, 1+q_len[i]+1:1+q_len[i]+1+d_len[i], :] for i in range(len(d_len))], batch_first=True)
        mask_p = self._create_mask_like(d_len, p_t_embed)
        mask_q = self._create_mask_like(q_len, q_t_embed)
        
        scores = self.ranker(q_t_embed, p_t_embed, mask_q, mask_p)

        logits = self.qa_outputs(p_t_embed)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_logits = start_logits*mask_p
        end_logits = end_logits*mask_p
        start_logits = nn.functional.log_softmax(start_logits, dim=1)
        end_logits = nn.functional.log_softmax(end_logits, dim=1)
        scores = nn.functional.log_softmax(scores)

        if start_positions is not None and end_positions is not None and is_selects is not None and has_answers is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index-1)
            end_positions.clamp_(0, ignored_index)
            for ind, s in enumerate(start_positions):
                if s.item() == ignored_index-1:
                    has_answers[ind] = 0.0

            # ignore negative RC loss
            mask = has_answers.unsqueeze(1).repeat(1, start_logits.size(1))
            start_logits = start_logits * mask
            end_logits = end_logits * mask
            scores = scores * has_answers

            def markTensor(labels, batch, length, end=False):
                pos_tensor=torch.zeros(batch, length).cuda()
                for i in range(len(labels)):
                    if end:
                        pos_tensor[i,labels[i].item()-1] = 1
                    else:
                        pos_tensor[i,labels[i].item()] = 1
                pos_tensor=torch.autograd.Variable(pos_tensor,requires_grad=False)
                return pos_tensor

            B, P = start_logits.size()
            starts = markTensor(start_positions, B, P)
            ends = markTensor(end_positions, B, P, end=True)
            selected_start_logits, start_ind = torch.min(starts*start_logits, dim=1)
            selected_end_logits, end_ind = torch.min(ends*end_logits, dim=1)
            
            total_loss = - torch.mean(selected_start_logits + selected_end_logits + scores)
            return total_loss
        else:
            return start_logits, end_logits, scores
