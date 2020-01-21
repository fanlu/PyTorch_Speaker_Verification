#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import json
import librosa
import argparse
import numpy as np
import concurrent.futures

from multiprocessing import cpu_count
from hparam import hparam as hp
from collections import defaultdict

# downloaded dataset path
audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))

def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved.
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")
    #os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(hp.data.test_path, exist_ok=True)    # make folder to save test file
    txt_path = glob.glob(os.path.join(hp.data.test_path_org, "lab/*.txt"))

    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    wav_paths = defaultdict(list)
    for txt in txt_path:
        txt_content = open(txt).readlines()
        txt_name = os.path.basename(txt).replace(".txt", "")
        for line in txt_content[1:]:
            try:
                no, content, real, role, gender, time = line.split("\t")
            except:
                print(txt, line)
                continue
            if real == "有效" and "顾客" in role:
                wav_paths["%s_%s_%s" % (txt_name, role.replace("顾客", "guke"), gender.replace("男", "male").replace("女", "female"))].append(os.path.join(hp.data.test_path_org, "segmented/%s_%s.wav" % (txt_name, no)))
    #print(wav_paths)

    #total_speaker_num = len(audio_path)
    total_speaker_num = len(wav_paths.keys())
    #train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("total speaker number : %d"%total_speaker_num)
    #print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))

    import pdb;pdb.set_trace()
    for i, (key, lst) in enumerate(wav_paths.items()):
        print("%dth speaker processing..."%i)
        utterances_spec = []
        for utter_name in lst:
            if utter_name[-4:] == '.wav':
                #utter_path = os.path.join(folder, utter_name)         # path of each utterance
                utter_path = utter_name
                utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
                intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection
                for interval in intervals:
                    if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                        S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance

        if not utterances_spec:
            continue
        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        #if i<train_speaker_num:      # save spectrogram as numpy file
        np.save(os.path.join(hp.data.test_path, "%s.npy"%key), utterances_spec)
        #else:
        #    np.save(os.path.join(hp.data.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)

def txt_2_dict(txt, min_tisv_frame):
    conversations, speakers = defaultdict(list), defaultdict(list)
    txt_content = open(txt).readlines()
    utter_min_len = (min_tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr
    txt_name = os.path.basename(txt).replace(".txt", "")
    for line in txt_content[1:]:
        try:
            no, content, real, role, gender, time = line.split("\t")
        except:
            print(txt, line)
            continue
        if real == "有效":
            f = os.path.join(os.path.dirname(txt).replace("lab", "segmented"), "%s_%s.wav" % (txt_name, no))
            utter, sr = librosa.core.load(f, hp.data.sr)
            intervals = librosa.effects.split(utter, top_db=30)
            add_dict = False
            for interval in intervals:
                if (interval[1]-interval[0]) > utter_min_len:
                    add_dict = True
                    break
            if add_dict and role:
                #conversations {key: [[u1, path],[u2, path],[u1, path], ...]}
                conversations[txt_name].append(("%s_%s_%s" % (txt_name, role.replace("客户", "guke").replace("顾客", "guke").replace("客服", "kefu"), gender.replace("男", "male").replace("女", "female")), os.path.join(os.path.dirname(txt).replace("lab", "segmented"), "%s_%s.wav" % (txt_name, no))))
                #wav_paths {u1: [path1, path2, ...]}
                speakers["%s_%s_%s" % (txt_name, role.replace("客户", "guke").replace("顾客", "guke").replace("客服", "kefu"), gender.replace("男", "male").replace("女", "female"))].append(os.path.join(os.path.dirname(txt).replace("lab", "segmented"), "%s_%s.wav" % (txt_name, no)))
    #print(wav_paths)
    return conversations, speakers

def generate_conversations(paths, save_dir, min_tisv_frame):
    conversations_json = open(os.path.join(save_dir, "conversations_json_%s.txt" % min_tisv_frame), "w")
    speakers_json = open(os.path.join(save_dir, "speakers_json_%s.txt" % min_tisv_frame), "w")
    all_txt = []
    for path in paths:
        txt_paths = glob.glob(os.path.join(path, "lab/*.txt"))
        print("%s:%s" % (len(txt_paths), path))
        for txt in txt_paths:
            all_txt.append(txt)
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count() - 5) as executor:
        future_to_f = {executor.submit(txt_2_dict, f, min_tisv_frame): f for f in all_txt}
        for future in concurrent.futures.as_completed(future_to_f):
            f = future_to_f[future]
            try:
                conversations, speakers = future.result()
                for i, (k, v) in enumerate(conversations.items()):
                    conversations_json.write(json.dumps({"conversation": k, "wav_files": v}) + "\n")
                for i, (k, v) in enumerate(speakers.items()):
                    speakers_json.write(json.dumps({"speaker": k, "wav_files": v}) + "\n")
            except Exception as exc:
                print('%r generated an exception: %s' % (f, exc))
    conversations_json.close()
    speakers_json.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configurations.')
    parser.add_argument('--min_tisv_frame', default=24, type=int, help='the minimum of tisv frame')
    parser.add_argument('--save_dir', default='train_tisv_1900h_json', type=str, help='json file to save')
    parser.add_argument('--origin_dir', nargs='*', help='dataset origin dir')
    args = parser.parse_args()
    #save_spectrogram_tisv()
    #generate_conversations(["/opt/cephfs1/asr/database/AM/kefu/kefu_500h/train", "/opt/cephfs1/asr/database/AM/kefu/kefu_200h/train", "/opt/cephfs1/asr/database/AM/kefu/kefu_122h/train", "/opt/cephfs1/asr/database/AM/kefu/kefu_179h/train", "/opt/cephfs1/asr/database/AM/kefu/kefu_500h_2/train", "/opt/cephfs1/asr/database/AM/kefu/kefu_400h"], args.save_dir, args.min_tisv_frame)
    #generate_conversations(["/opt/cephfs1/asr/database/AM/kefu/kefu_200h/test/dev/"], "test_tisv_200h/conversations_json_2.txt", "test_tisv_200h/speakers_json_2.txt")
    generate_conversations(args.origin_dir, args.save_dir, args.min_tisv_frame)
    #generate_speakers(["/opt/cephfs1/asr/database/AM/kefu/kefu_500h/train", "/opt/cephfs1/asr/database/AM/kefu/kefu_200h/train", "/opt/cephfs1/asr/database/AM/kefu/kefu_122h/train", "/opt/cephfs1/asr/database/AM/kefu/kefu_179h/train", "/opt/cephfs1/asr/database/AM/kefu/kefu_500h_2/train", "/opt/cephfs1/asr/database/AM/kefu/kefu_400h"], "train_tisv_1900h_json/conversations_json_2.txt")
