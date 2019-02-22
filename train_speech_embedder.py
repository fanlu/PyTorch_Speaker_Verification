#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: harry
"""

import os
import glob
import random
import time
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed, KefuDataset, collate_fn
from speech_embedder_net import SpeechEmbedder, SpeechEmbedder2, GE2ELoss, get_centroids, get_cossim

def train(model_path, train_path_json, min_tisv_frame, max_tisv_frame, ckpt_dir, bidirectional=False, dropout=0.0):
    writer = SummaryWriter()
    #if hp.data.data_preprocessed:
    #    train_dataset = SpeakerDatasetTIMITPreprocessed()
    #else:
    #    train_dataset = SpeakerDatasetTIMIT()
    #train_dataset = KefuDataset([hp.data.train_path_org])
    train_dataset = KefuDataset(train_path_json, hp.train.M, min_tisv_frame=min_tisv_frame, max_tisv_frame=max_tisv_frame)
    embedder_net = SpeechEmbedder2(bidirectional=bidirectional, dropout=dropout)
    ge2e_loss = GE2ELoss()
    #import pdb;pdb.set_trace()
    if hp.train.restore:
        embedder_net.load_state_dict(torch.load(model_path))

    if hp.ngpu >= 1:
        embedder_net = torch.nn.DataParallel(embedder_net.cuda(), device_ids=list(range(hp.ngpu)))
        ge2e_loss = torch.nn.DataParallel(ge2e_loss.cuda(), device_ids=list(range(hp.ngpu)))
        hp.train.N *= hp.ngpu
    #train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True, collate_fn=collate_fn)

    device = torch.device("cuda" if hp.ngpu > 0 else "cpu")
    # embedder_net = embedder_net.to(device)
    #ge2e_loss = ge2e_loss.to(device)
    #Both ne
    optimizer = torch.optim.SGD([
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], lr=hp.train.lr)
    scheduler = MultiStepLR(optimizer, milestones=hp.train.lr_schedule, gamma=0.1)

    os.makedirs(ckpt_dir, exist_ok=True)

    embedder_net.train()
    iteration = 0
    #import pdb;pdb.set_trace()
    for e in range(hp.train.epochs):
        scheduler.step()
        print("epochs:%s, lr is:%s" % (e, optimizer.param_groups[0]['lr']))
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iteration)
        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader):
            seqs, seq_lengths, seq_keys = mel_db_batch
            mel_db_batch, seq_lengths = seqs.to(device), seq_lengths.to(device)

            #mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            #perm = random.sample(range(0, hp.train.N*hp.train.M), hp.train.N*hp.train.M)
            #unperm = list(perm)
            #for i,j in enumerate(perm):
            #    unperm[j] = i
            #mel_db_batch = mel_db_batch[perm]
            #gradient accumulates
            optimizer.zero_grad()

            embeddings = embedder_net(mel_db_batch, seq_lengths)
            #embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))

            #get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings) #wants (Speaker, Utterances, embedding)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()

            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t".format(time.ctime(), e+1,
                        batch_id+1, len(train_dataset)//hp.train.N, iteration,loss, total_loss / (batch_id + 1))
                writer.add_scalar('train/loss', loss, iteration)
                writer.add_scalar('train/total_loss', total_loss/(batch_id+1), iteration)
                print(mesg)
                if hp.train.log_file is not None:
                    with open(os.path.join(ckpt_dir, "Stats"),'a') as f:
                        f.write(mesg + "\n")

        if ckpt_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            #embedder_net.eval().cpu()
            #import pdb;pdb.set_trace()
            ckpt_model_filename = "ckpt_epoch_" + str(e+1) + "_batch_id_" + str(batch_id+1) + ".pth"
            ckpt_model_path = os.path.join(ckpt_dir, ckpt_model_filename)
            torch.save(embedder_net.module.state_dict(), ckpt_model_path)
            avg_ERR = test(embedder_net.module)
            writer.add_scalar("test/avg_ERR", avg_ERR, iteration)
            embedder_net.train()

    #save model
    embedder_net.eval().cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
    save_model_path = os.path.join(ckpt_dir, save_model_filename)
    torch.save(embedder_net.module.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)

def test(model, test_path_json=hp.data.test_path_json, min_tisv_frame=hp.data.min_tisv_frame, max_tisv_frame=hp.data.tisv_frame):

    #if hp.data.data_preprocessed:
    #    test_dataset = SpeakerDatasetTIMITPreprocessed()
    #else:
    #    test_dataset = SpeakerDatasetTIMIT()
    test_dataset = KefuDataset(test_path_json, hp.test.M, min_tisv_frame=min_tisv_frame, max_tisv_frame=max_tisv_frame)
    #test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True, collate_fn=collate_fn)

    if not isinstance(model, SpeechEmbedder2):
        embedder_net = SpeechEmbedder2()
        embedder_net.load_state_dict(torch.load(model))
    else:
        embedder_net = model
    embedder_net.cuda()
    embedder_net.eval()

    avg_EER = 0
    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, mel_db_batch in enumerate(test_loader):
            seqs, seq_lengths, seq_keys = mel_db_batch
            mel_db_batch, seq_lengths = seqs.cuda().reshape(hp.test.N, hp.test.M, seqs.size(1), seqs.size(2)), seq_lengths.cuda().reshape(hp.test.N, hp.test.M)
            #import pdb;pdb.set_trace()
            assert hp.test.M % 2 == 0
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
            enrollment_length, verification_length = torch.split(seq_lengths, int(mel_db_batch.size(1)/2), dim=1)

            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M//2, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (hp.test.N*hp.test.M//2, verification_batch.size(2), verification_batch.size(3)))

            enrollment_length = enrollment_length.flatten()
            verification_length = verification_length.flatten()

            #perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
            #unperm = list(perm)
            #for i,j in enumerate(perm):
            #    unperm[j] = i

            #verification_batch = verification_batch[perm]
            enrollment_embeddings = embedder_net(enrollment_batch, enrollment_length)
            verification_embeddings = embedder_net(verification_batch, verification_length)

            #verification_embeddings = verification_embeddings[unperm]

            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))

            enrollment_centroids = get_centroids(enrollment_embeddings)

            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0

            for thres in [0.01*i+0.5 for i in range(50)]:
                sim_matrix_thresh = sim_matrix>thres

                FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                /(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)

                FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                /(float(hp.test.M/2))/hp.test.N)

                # Save threshold when FAR = FRR (=EER)
                if diff> abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            #print("EER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR))
        avg_EER += batch_avg_EER/(batch_id+1)
    avg_EER = avg_EER / hp.test.epochs
    print("EER across {0} epochs: {1:.4f}, model: {2}".format(hp.test.epochs, avg_EER, model))
    return avg_EER

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='configurations.')
    parser.add_argument('--ckpt_dir', default=hp.train.checkpoint_dir, type=str, help='checkpoint save dir')
    parser.add_argument('--train_path_json', default=hp.data.train_path_json, type=str, help='the data training input')
    parser.add_argument('--test_path_json', default=hp.data.test_path_json, type=str, help='the data test input')
    parser.add_argument('--min_tisv_frame', default=hp.data.min_tisv_frame, type=int, help='the minimum of tisv frame')
    parser.add_argument('--max_tisv_frame', default=hp.data.tisv_frame, type=int, help='the maxmum of tisv frame')
    parser.add_argument('--training', default=True, type=bool, help='is training')
    parser.add_argument('--dropout', default=0.0, type=float, help='the dropout of lstm')
    parser.add_argument('--bidirectional', default=False, type=bool, help='the bidirectional of lstm')
    args = parser.parse_args()
    if args.training:
        train(hp.model.model_path, args.train_path_json, args.min_tisv_frame, args.max_tisv_frame, args.ckpt_dir, bidirectional=args.bidirectional, dropout=args.dropout)
    else:
        for i in glob.glob("%s/ckpt*_[1-9]0*" % args.ckpt_dir):
            test(i, args.test_path_json, args.min_tisv_frame, args.max_tisv_frame)
