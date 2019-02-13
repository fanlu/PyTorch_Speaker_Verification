#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import glob
import numpy as np
import json
import os
import librosa
import random
from random import shuffle
import torch
from torch.utils.data import Dataset
from collections import defaultdict

from hparam import hparam as hp
from utils import mfccs_and_spec, filter_bank

class SpeakerDatasetTIMIT(Dataset):

    def __init__(self):

        if hp.training:
            self.path = hp.data.train_path_unprocessed
            self.utterance_number = hp.train.M
        else:
            self.path = hp.data.test_path_unprocessed
            self.utterance_number = hp.test.M
        self.speakers = glob.glob(os.path.dirname(self.path))
        shuffle(self.speakers)

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):

        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker+'/*.wav')
        shuffle(wav_files)
        wav_files = wav_files[0:self.utterance_number]

        mel_dbs = []
        for f in wav_files:
            _, mel_db, _ = mfccs_and_spec(f, wav_process = True)
            mel_dbs.append(mel_db)
        return torch.Tensor(mel_dbs)

class SpeakerDatasetTIMITPreprocessed(Dataset):

    def __init__(self, shuffle=True, utter_start=0):

        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = os.listdir(self.path)
        self.shuffle=shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        np_file_list = os.listdir(self.path)

        if self.shuffle:
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = np_file_list[idx]
        #print(self.path, selected_file)
        utters = np.load(os.path.join(self.path, selected_file))        # load utterance spectrogram of selected speaker
        #import pdb;pdb.set_trace()
        if self.shuffle:
            #print(utters.shape[0], self.utter_num)
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
            utterance = utters[utter_index]
        else:
            utterance = utters[self.utter_start: self.utter_start+self.utter_num] # utterances of a speaker [batch(M), n_mels, frames]

        utterance = utterance[:,:,:160]               # TODO implement variable length batch size

        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
        return utterance

class KefuDataset(Dataset):
    def __init__(self, path, m, shuffle=True):
        #if hp.training:
        #    #self.path = hp.data.train_path_unprocessed
        #    self.utterance_number = hp.train.M
        #else:
        #    #self.path = hp.data.test_path_unprocessed
        #    self.utterance_number = hp.test.M
        self.utterance_number = m
        self.speakers_list = []
        f = open(path, "r")
        for line in f.readlines():
            dic = json.loads(line.strip())
            self.speakers_list.append((dic.get("speaker"), dic.get("wav_files")))
        self.shuffle = shuffle

    def __len__(self):
        return len(self.speakers_list)

    def __getitem__(self, idx):
        #if self.shuffle:
        #    key, wav_files = random.sample(self.speakers_list, 1)[0]
        #else:
        key, wav_files = self.speakers_list[idx]
        if len(wav_files) > self.utterance_number:
            utter_index = np.array(random.sample(range(len(wav_files)), self.utterance_number))
        else:
            utter_index = np.random.randint(0, len(wav_files), self.utterance_number)
        #shuffle(wav_files)
        #wav_files = wav_files[0:self.utterance_number]
        #print(utter_index)
        wav_files = np.array(wav_files)[utter_index]
        mel_dbs = []
        for f in wav_files:
            #_, mel_db, _ = mfccs_and_spec(f, wav_process = True)
            mel_db = filter_bank(f)
            if mel_db:
                mel_db = random.sample(mel_db, 1)[0]
                frames = mel_db.shape[1]
                start = np.random.randint(0, frames-24)
                end = np.random.randint(start+24, min(frames, start+160))
                mel_dbs.append(np.transpose(mel_db[:, start:end], axes=(1, 0)))
            else:
                print("wav_file:%s" % f)
        if len(mel_dbs) < self.utterance_number:
            mel_dbs_index = np.random.randint(0, len(mel_dbs), self.utterance_number)
            mel_dbs = np.array(mel_dbs)[mel_dbs_index]
        return {"key": key, "mel_dbs": mel_dbs}

#https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py
def collate_fn(data):
    def merge(sequences):
        lengths = [seq['mel_db'].shape[0] for seq in sequences]
        keys = [seq['key'] for seq in sequences]
        padded_seqs = np.zeros((len(sequences), max(lengths), sequences[0]['mel_db'].shape[1]), dtype=np.float32)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end, :] = seq['mel_db'][:end, :]
        return padded_seqs, lengths, keys
    data2 = []
    for d in data:
        for mel_db in d['mel_dbs']:
            data2.append({"key": d['key'], "mel_db": mel_db})
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    # data2.sort(key=lambda x: x['mel_db'].shape[0], reverse=True)

    seqs, seq_lengths, seq_keys = merge(data2)

    return torch.tensor(seqs), torch.tensor(seq_lengths), seq_keys


