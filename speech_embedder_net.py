#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:58:34 2018

@author: harry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hparam import hparam as hp
from utils import get_centroids, get_cossim, get_cossim2, calc_loss, calc_loss2

class SpeechEmbedder(nn.Module):

    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)

    def forward(self, x):
        #self.LSTM_stack.flatten_parameters()
        x, _ = self.LSTM_stack(x.float()) #(batch, frames, n_mels)
        #print("1", x.shape)
        #only use last frame
        x = x[:,x.size(1)-1]
        #print("2", x.shape)
        x = self.projection(x.float())
        #print("3", x.shape)
        x = F.normalize(x, dim=-1, eps=1e-6)
        #print("4", torch.norm(x).shape)
        #print("--------", x.shape)
        return x

class SpeechEmbedder2(nn.Module):

    def __init__(self, bidirectional=False, dropout=0.0):
        super(SpeechEmbedder2, self).__init__()
        self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        for name, param in self.LSTM_stack.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)
        self.bidirectional = bidirectional
        if self.bidirectional:
            hp.model.hidden *= 2
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)

    def forward(self, x, x_len):
        #import pdb;pdb.set_trace()
        _, idx_sort = torch.sort(x_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        input_x = torch.index_select(x, 0, idx_sort)
        length_list = list(x_len[idx_sort])
        pack = torch.nn.utils.rnn.pack_padded_sequence(input_x, length_list, batch_first=True)
        #self.LSTM_stack.flatten_parameters()
        x, (ht, ct) = self.LSTM_stack(pack) #(batch, frames, n_mels)
        #if isinstance(x, PackedSequence):
        #    x = x[0]
        #    out = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.bidirectional:
            output = torch.cat((ht[-1], ht[-2]), dim=-1)
        else:
            output = ht[-1]
        output = output[idx_unsort]
        #print("1", x.shape)
        #only use last frame
        #x = x[:,x.size(1)-1]
        #print("2", x.shape)
        x = self.projection(output)
        #print("3", x.shape)
        x = F.normalize(x, dim=-1, eps=1e-6)
        #print("4", torch.norm(x).shape)
        #print("--------", x.shape)
        return x

class GE2ELoss(nn.Module):

    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True)
        # self.device = device

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        # centroids = get_centroids(embeddings)
        # cossim = get_cossim(embeddings, centroids)
        # embeddings = F.normalize(embeddings, dim=-1, eps=1e-6)
        cossim = get_cossim2(embeddings, None)
        #print("cossim.shape is :", cossim.shape)
        sim_matrix = self.w*cossim + self.b
        # loss, per_embedding_loss = calc_loss(sim_matrix)
        per_embedding_loss = calc_loss2(sim_matrix)
        #print("=====", per_embedding_loss.shape)
        return per_embedding_loss
