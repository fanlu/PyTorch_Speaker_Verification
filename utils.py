#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:56:19 2018

@author: harry
"""
import librosa
import numpy as np
import torch
import torch.autograd as grad
import torch.nn.functional as F

from hparam import hparam as hp

def get_centroids(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid = centroid + utterance
        centroid = centroid/len(speaker)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids

def get_centroid(embeddings, speaker_num, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid/(len(embeddings[speaker_num])-1)
    return centroid

def get_cossim(embeddings, centroids):
    # Calculates cosine similarity matrix. Requires (N, M, feature) input
    cossim = torch.zeros(embeddings.size(0),embeddings.size(1),centroids.size(0))
    for speaker_num, speaker in enumerate(embeddings):
        for utterance_num, utterance in enumerate(speaker):
            for centroid_num, centroid in enumerate(centroids):
                if speaker_num == centroid_num:
                    centroid = get_centroid(embeddings, speaker_num, utterance_num)
                output = F.cosine_similarity(utterance,centroid,dim=0)+1e-6
                cossim[speaker_num][utterance_num][centroid_num] = output
    return cossim

def get_cossim2(embeddings, centroids=None):
    N, M, feature = embeddings.shape
    if not centroids:
        center = torch.mean(embeddings, 1)
        center = F.normalize(center, dim=-1, eps=1e-6)

        center_except = torch.reshape(torch.sum(embeddings, 1, keepdim=True) - embeddings, (N*M, feature))
        center_except = F.normalize(center_except, dim=-1, eps=1e-6)

        cossim = torch.cat([torch.cat([torch.sum(center_except[i*M:(i+1)*M, :]*embeddings[j, :, :], 1, keepdim=True) if i==j else torch.sum(center[i:(i+1),:]*embeddings[j,:,:], 1, keepdim=True) for i in range(N)], 1) for j in range(N)], 0)
        return cossim

        # torch.cat([torch.cat([torch.sum(center_except[i*5:(i+1)*5, :]*embeddings[j, :, :], 1, keepdim=True) if i==j else torch.sum(center[i:(i+1),:]*embeddings[j,:,:], 1, keepdim=True) for i in range(32)], 1) for j in range(32)], 0)


def calc_loss(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))
    for j in range(len(sim_matrix)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum()+1e-6).log_()))
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss

def calc_loss2(sim_matrix):
    NM, N = sim_matrix.shape
    M = int(NM / N)
    S_correct = torch.cat([sim_matrix[i*M:(i+1)*M, i:(i+1)] for i in range(N)], 0)
    loss = -torch.sum(S_correct - torch.log(torch.sum(torch.exp(sim_matrix), 1, keepdim=True) + 1e-6), 0, keepdim=True)
    return loss

def normalize_0_1(values, max_value, min_value):
    normalized = np.clip((values - min_value) / (max_value - min_value), 0, 1)
    return normalized

def mfccs_and_spec(wav_file, wav_process = False, calc_mfccs=False, calc_mag_db=False):
    sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
    window_length = int(hp.data.window*hp.data.sr)
    hop_length = int(hp.data.hop*hp.data.sr)
    duration = hp.data.tisv_frame * hp.data.hop + hp.data.window

    # Cut silence and fix length
    if wav_process == True:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(hp.data.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)

    spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)

    mel_basis = librosa.filters.mel(hp.data.sr, hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, mag_spec)

    mag_db = librosa.amplitude_to_db(mag_spec)
    #db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T

    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T

    return mfccs, mel_db, mag_db

def filter_bank(wav_file):
    utter_min_len = (24 * hp.data.hop + hp.data.window) * hp.data.sr
    utter, sr = librosa.core.load(wav_file, hp.data.sr)
    intervals = librosa.effects.split(utter, top_db=30)
    utterances_spec = []
    for interval in intervals:
        if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
            utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
            S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                  win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
            S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
            utterances_spec.append(S)
    return utterances_spec

if __name__ == "__main__":
    w = grad.Variable(torch.tensor(1.0))
    b = grad.Variable(torch.tensor(0.0))
    embeddings = torch.tensor([[0,1,0],[0,0,1], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]).to(torch.float).reshape(3,2,3)
    centroids = get_centroids(embeddings)
    cossim = get_cossim(embeddings, centroids)
    sim_matrix = w*cossim + b
    loss, per_embedding_loss = calc_loss(sim_matrix)
