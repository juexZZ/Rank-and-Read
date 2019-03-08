'''
Ranker:
Neural IR method: conv-KNRM
'''
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from util.utils import kernal_mus, kernel_sigmas


class cknrm(nn.Module):
    """
    kernel pooling layer
    """

    def __init__(self, num_kernels, rembed_dim, ifcuda):
        """

        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(cknrm, self).__init__()
        tensor_mu = torch.FloatTensor(kernal_mus(num_kernels))
        tensor_sigma = torch.FloatTensor(kernel_sigmas(num_kernels))
        if ifcuda and torch.cuda.is_available():
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()   

        self.d_word_vec = rembed_dim
        print('ranker_embed_dim: ', self.d_word_vec)
        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, num_kernels)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, num_kernels)
        #self.wrd_emb = ranker_embedder
        self.dense_f = nn.Linear(num_kernels * 9, 1, 1)
        self.tanh = nn.Tanh()
        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, 128, (1, rembed_dim)),
            nn.ReLU()
        )

        self.conv_bi = nn.Sequential(
            nn.Conv2d(1, 128, (2, rembed_dim)),
            nn.ReLU()
        )
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, 128, (3, rembed_dim)),
            nn.ReLU()
        )


    def get_intersect_matrix(self, q_embed, d_embed, atten_q, atten_d):

        sim = torch.bmm(q_embed, d_embed).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
        #np.save('/data/disk4/private/zhangjuexiao/e2e_case/rank_case'+"_"+str(q_embed.size()[1])+"_"+str(d_embed.size()[2]), sim.data.cpu().numpy())
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * atten_d.type_as(self.sigma)
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * atten_q.type_as(self.sigma)
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum



    def forward(self, qw_embed, dw_embed, inputs_qwm, inputs_dwm):
        qwu_embed = torch.transpose(torch.squeeze(self.conv_uni(qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 0.000000001
        qwb_embed = torch.transpose(torch.squeeze(self.conv_bi (qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 0.000000001
        qwt_embed = torch.transpose(torch.squeeze(self.conv_tri(qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 0.000000001
        dwu_embed = torch.squeeze(self.conv_uni(dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 0.000000001
        dwb_embed = torch.squeeze(self.conv_bi (dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 0.000000001
        dwt_embed = torch.squeeze(self.conv_tri(dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 0.000000001
        qwu_embed_norm = F.normalize(qwu_embed, p=2, dim=2, eps=1e-10)
        qwb_embed_norm = F.normalize(qwb_embed, p=2, dim=2, eps=1e-10)
        qwt_embed_norm = F.normalize(qwt_embed, p=2, dim=2, eps=1e-10)
        dwu_embed_norm = F.normalize(dwu_embed, p=2, dim=1, eps=1e-10)
        dwb_embed_norm = F.normalize(dwb_embed, p=2, dim=1, eps=1e-10)
        dwt_embed_norm = F.normalize(dwt_embed, p=2, dim=1, eps=1e-10)
        mask_qw = inputs_qwm.view(qw_embed.size()[0], qw_embed.size()[1], 1)
        mask_dw = inputs_dwm.view(dw_embed.size()[0], 1, dw_embed.size()[1], 1)
        mask_qwu = mask_qw[:, :qw_embed.size()[1] - (1 - 1), :]
        mask_qwb = mask_qw[:, :qw_embed.size()[1] - (2 - 1), :]
        mask_qwt = mask_qw[:, :qw_embed.size()[1] - (3 - 1), :]
        mask_dwu = mask_dw[:, :, :dw_embed.size()[1] - (1 - 1), :]
        mask_dwb = mask_dw[:, :, :dw_embed.size()[1] - (2 - 1), :]
        mask_dwt = mask_dw[:, :, :dw_embed.size()[1] - (3 - 1), :]
        log_pooling_sum_wwuu = self.get_intersect_matrix(qwu_embed_norm, dwu_embed_norm, mask_qwu, mask_dwu)
        log_pooling_sum_wwut = self.get_intersect_matrix(qwu_embed_norm, dwt_embed_norm, mask_qwu, mask_dwt)
        log_pooling_sum_wwub = self.get_intersect_matrix(qwu_embed_norm, dwb_embed_norm, mask_qwu, mask_dwb)
        log_pooling_sum_wwbu = self.get_intersect_matrix(qwb_embed_norm, dwu_embed_norm, mask_qwb, mask_dwu)
        log_pooling_sum_wwtu = self.get_intersect_matrix(qwt_embed_norm, dwu_embed_norm, mask_qwt, mask_dwu)
        log_pooling_sum_wwbb = self.get_intersect_matrix(qwb_embed_norm, dwb_embed_norm, mask_qwb, mask_dwb)
        log_pooling_sum_wwbt = self.get_intersect_matrix(qwb_embed_norm, dwt_embed_norm, mask_qwb, mask_dwt)
        log_pooling_sum_wwtb = self.get_intersect_matrix(qwt_embed_norm, dwb_embed_norm, mask_qwt, mask_dwb)
        log_pooling_sum_wwtt = self.get_intersect_matrix(qwt_embed_norm, dwt_embed_norm, mask_qwt, mask_dwt)
        
        log_pooling_sum = torch.cat([ log_pooling_sum_wwuu, log_pooling_sum_wwut, log_pooling_sum_wwub, log_pooling_sum_wwbu, log_pooling_sum_wwtu,\
            log_pooling_sum_wwbb, log_pooling_sum_wwbt, log_pooling_sum_wwtb, log_pooling_sum_wwtt], 1)
        output = torch.squeeze(self.dense_f(log_pooling_sum), 1)
        return output


class knrm(nn.Module):
    """
    kernel pooling layer
    """

    def __init__(self, num_kernels, rembed_dim, ifcuda):
        """

        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(knrm, self).__init__()

        tensor_mu = torch.FloatTensor(kernal_mus(num_kernels))
        tensor_sigma = torch.FloatTensor(kernel_sigmas(num_kernels))
        if ifcuda and torch.cuda.is_available():
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
        self.mu = Variable(tensor_mu, requires_grad = False).view(1, 1, 1, num_kernels)
        self.sigma = Variable(tensor_sigma, requires_grad = False).view(1, 1, 1, num_kernels)
        self.dense = nn.Linear(num_kernels, 1, 1)

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):

        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1) # n*m*d*1
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d.type_as(self.sigma)
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q.type_as(self.sigma) * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 1)#soft-TF 
        return log_pooling_sum


    def forward(self, q_embed, d_embed, mask_q, mask_d):
        q_embed_norm = F.normalize(q_embed, 2, 2)
        d_embed_norm = F.normalize(d_embed, 2, 2)
        mask_d = mask_d.view(mask_d.size()[0], 1, mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)
        log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q, mask_d)
        output = torch.squeeze(self.dense(log_pooling_sum), 1)
        return output
