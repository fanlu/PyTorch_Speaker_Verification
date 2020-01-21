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
        self.projection_hidden = hp.model.hidden * 2 if self.bidirectional else hp.model.hidden
        self.projection = nn.Linear(self.projection_hidden, hp.model.proj)

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

class TDNNStack(nn.Module):
    def __init__(self,input_dim, nndef, dropout, use_SE=False, SE_ratio=4, use_selu=False):
        super(TDNNStack, self).__init__()
        self.input_dim = input_dim
        model_list = OrderedDict()
        ly = 0
        for item in nndef.split("."):
            out_dim, k_size, dilation = [ int(x) for x in item.split("_")]
            model_list["TDNN%d"%ly] = nn.Conv1d(input_dim, out_dim, k_size, dilation=dilation)
            if use_selu:
                model_list["SeLU%d"%ly] = nn.SELU()
            else:
                model_list["ReLU%d"%ly] = nn.ReLU()
                model_list["batch_norm%d"%ly] = nn.BatchNorm1d(out_dim)
            if use_SE:
                model_list["SEnet%d"%ly] = SquExiNet(out_dim, SE_ratio)
            if dropout != 0.0:
                model_list['dropout%d'%ly] = nn.Dropout(dropout)
            input_dim = out_dim
            ly = ly + 1
        self.model = nn.Sequential(model_list)
        self.output_dim = input_dim

    def forward(self, input):
        #input: seqLength X batchSize X dim
        #output: seqLength X batchSize X dim (similar to lstm)
        input = input.contiguous().transpose(0,1).transpose(1,2) #batchSize, dim, seqLength
        output = self.model(input)
        output = output.contiguous().transpose(1,2).transpose(0,1) #seqLength, batchSize, dim
        return output

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

class localatt(nn.Module):
    def __init__(self, featdim, nhid, ncell, nout):
        super(localatt, self).__init__()

        self.featdim = featdim
        self.nhid = nhid
        self.fc1 = nn.Linear(featdim, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.do2 = nn.Dropout()


        self.blstm = tc.nn.LSTM(nhid, ncell, 1,
                batch_first=True,
                dropout=0.5,
                bias=True,
                bidirectional=True)

        self.u = nn.Parameter(tc.zeros((ncell*2,)))
        # self.u = Variable(tc.zeros((ncell*2,)))

        self.fc3 = nn.Linear(ncell*2, nout)

        self.apply(init_linear)

    def forward(self, inputs_lens_tuple):

        inputs = Variable(inputs_lens_tuple[0])
        batch_size = inputs.size()[0]
        lens = list(inputs_lens_tuple[1])

        indep_feats = inputs.view(-1, self.featdim) # reshape(batch)

        indep_feats = F.relu(self.fc1(indep_feats))

        indep_feats = F.relu(self.do2(self.fc2(indep_feats)))

        batched_feats = indep_feats.view(batch_size, -1, self.nhid)

        packed = pack_padded_sequence(batched_feats, lens, batch_first=True)

        output, hn = self.blstm(packed)

        padded, lens = pad_packed_sequence(output, batch_first=True, padding_value=0.0)

        alpha = F.softmax(tc.matmul(padded, self.u))

        return F.softmax((self.fc3(tc.sum(tc.matmul(alpha, padded), dim=1))))

