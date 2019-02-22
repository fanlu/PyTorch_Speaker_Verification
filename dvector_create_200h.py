#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:34:01 2018

@author: Harry

Creates "segment level d vector embeddings" compatible with
https://github.com/google/uis-rnn

"""

import glob
import librosa
import numpy as np
import os
import torch
import json
import argparse

from collections import defaultdict
from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk


def concat_segs(times, segs):
    #Concatenate continuous voiced segments
    concat_seg = []
    seg_concat = segs[0]
    for i in range(0, len(times)-1):
        if times[i][1] == times[i+1][0]:
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
        else:
            concat_seg.append(seg_concat)
            seg_concat = segs[i+1]
    else:
        concat_seg.append(seg_concat)
    return concat_seg

def get_STFTs(segs):
    #Get 240ms STFT windows with 50% overlap
    sr = hp.data.sr
    STFT_frames = []
    for seg in segs:
        S = librosa.core.stft(y=seg, n_fft=hp.data.nfft,
                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
        S = np.abs(S)**2
        mel_basis = librosa.filters.mel(sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
        for j in range(0, S.shape[1], int(.12/hp.data.hop)):
            if j + 24 < S.shape[1]:
                STFT_frames.append(S[:,j:j+24])
            else:
                break
    return STFT_frames

def align_embeddings(embeddings):
    partitions = []
    start = 0
    end = 0
    j = 1
    for i, embedding in enumerate(embeddings):
        if (i*.12)+.24 < j*.401:
            end = end + 1
        else:
            partitions.append((start,end))
            start = end
            end = end + 1
            j += 1
    else:
        partitions.append((start,end))
    avg_embeddings = np.zeros((len(partitions),256))
    for i, partition in enumerate(partitions):
        avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0)
    return avg_embeddings

def generate_dvector(path, save_path, model_path):
    #dataset path
    txt_path = glob.glob(os.path.join(path, "lab/*.txt"))
    wav_paths = []
    txt_wavs = defaultdict(list)
    for txt in txt_path:
        txt_content = open(txt).readlines()
        txt_name = os.path.basename(txt).replace(".txt", "")
        for line in txt_content[1:]:
            try:
                no, content, real, role, gender, time = line.split("\t")
            except:
                print(txt, line)
                continue
            if real == "有效":
                txt_wavs[txt_name].append(("%s_%s" % (txt_name, role.replace("顾客", "guke").replace("客服", "kefu")), os.path.join(path, "segmented/%s_%s.wav" % (txt_name, no))))
    #audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))

    # total_speaker_num = len(audio_path)
    # train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test

    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.cuda()
    embedder_net.eval()
    conversations_json = open('./train_tisv_1900h_json/conversations_json_2.txt').readlines()
    count = 0
    # for i, folder in enumerate(audio_path):
    #for i, (key, value) in enumerate(txt_wavs.items()):
    for i, line in enumerate(conversations_json):
        line_json = json.loads(line)
        key = line_json['conversation']
        value = line_json['wav_files']
        train_sequence = []
        train_cluster_id = []
        #import pdb;pdb.set_trace()
        for (k, v) in value:
            if v[-4:] == '.wav':
                times, segs = VAD_chunk(2, v)
                if segs == []:
                    print('No voice activity detected')
                    continue
                concat_seg = concat_segs(times, segs)
                STFT_frames = get_STFTs(concat_seg)
                if not STFT_frames:
                    continue
                STFT_frames = np.stack(STFT_frames, axis=2)
                STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0))).cuda()
                embeddings = embedder_net(STFT_frames)
                aligned_embeddings = align_embeddings(embeddings.cpu().detach().numpy())
                train_sequence.append(aligned_embeddings)
                for embedding in aligned_embeddings:
                    train_cluster_id.append(str(k))
                count = count + 1
        if (i+1) % 100 == 0:
            print('Processed {0}/{1} files'.format(i, len(conversations_json)))

        train_sequence = np.concatenate(train_sequence,axis=0)
        train_cluster_id = np.asarray(train_cluster_id)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.save('%s_sequence_%s' % (save_path, key), train_sequence)
        np.save('%s_cluster_id_%s' % (save_path, key), train_cluster_id)

if __name__ == "__main__":
    import pdb;pdb.set_trace()
    parser = argparse.ArgumentParser(description='configurations.')
    parser.add_argument('--save_dir', default='', type=str, help='dvector save dir')
    parser.add_argument('--model', default=hp.model.model_path, type=str, help='which model will be load')
    args = parser.parse_args()
    if args.save_dir:
        generate_dvector(hp.data.train_path_org, os.path.join(args.save_dir, "train"), args.model) #'dvector_data_1900h_12_768.256_0.01_24_40_epoch_80_conversations2/train')
        generate_dvector(hp.data.test_path_org, os.path.join(args.save_dir, "test"), args.model) #'dvector_data_1900h_12_768.256_0.01_24_40_epoch_80_conversations2/test')

