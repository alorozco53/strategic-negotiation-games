# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pdb
import random
import re
import os
import time

from threading import Thread
import pandas as pd
import numpy as np
import torch
from torch import optim
from torch import autograd
from tqdm import tqdm

import torch.nn as nn
import data
import utils
from utils import ContextGenerator
from agent import RnnAgent, RnnRolloutAgent, RlAgent, HierarchicalAgent
from dialog import Dialog, DialogLogger, TripleDialog, TwoVsOneDialog
from selfplay import get_agent_type
from domain import get_domain

torch.autograd.set_detect_anomaly(True)

class Reinforce(object):
    def __init__(self,
                 dialog,
                 ctx_gen,
                 args,
                 engine,
                 corpus,
                 logger=None,
                 tqdm=None):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.engine = engine
        self.corpus = corpus
        self.tqdm = tqdm
        self.logger = logger if logger else DialogLogger()

    def run(self):
        validset, validset_stats = self.corpus.valid_dataset(self.args.bsz)
        trainset, trainset_stats = self.corpus.train_dataset(self.args.bsz)

        n = 0
        loop = self.ctx_gen.iter(self.args.nepoch)
        total = self.args.nepoch * (self.args.ctx_num or len(self.ctx_gen.ctxs))
        if self.tqdm is not None:
            loop = self.tqdm(loop, total=total)
        for ctxs in loop:
            n += 1
            if n > total:
                break
            if self.args.engine_train and \
               self.args.sv_train_freq > 0 and \
               n % self.args.sv_train_freq == 0:
                batch = random.choice(trainset)
                self.engine.model.train()
                self.engine.train_batch(batch)
                # self.engine.model.eval()

            self.logger.dump('=' * 80)
            self.dialog.run(ctxs, self.logger)
            self.logger.dump('=' * 80)
            self.logger.dump('')
            if n % 100 == 0:
                self.logger.dump('%d: %s' % (n, self.dialog.show_metrics()),
                                 forced=True)
            # input()

        def dump_stats(dataset, stats, name):
            loss, select_loss = self.engine.valid_pass(dataset, stats)
            self.logger.dump('final: %s_loss %.3f %s_ppl %.3f' % (
                name, float(loss), name, np.exp(float(loss))),
                forced=True)
            self.logger.dump('final: %s_select_loss %.3f %s_select_ppl %.3f' % (
                name, float(select_loss), name, np.exp(float(select_loss))),
                forced=True)

        #dump_stats(trainset, trainset_stats, 'train')
        #dump_stats(validset, validset_stats, 'valid')

        self.logger.dump('final: %s' % self.dialog.show_metrics(), forced=True)

class RLThread(Thread):
    def __init__(self, args, seed, dialog, ctx_gen, engine, corpus, logger, tqdm):
        super(RLThread, self).__init__()
        self.seed = seed
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.engine = engine
        self.corpus = corpus
        self.logger = logger
        self.tqdm = tqdm
        
    def run(self):
        self.args.seed = self.seed
        utils.set_seed(self.args.seed)
        reinforce = Reinforce(self.dialog, self.ctx_gen, self.args,
                              self.engine, self.corpus, self.logger, self.tqdm)
        reinforce.run()
        rews = pd.DataFrame({b.name: b.all_rewards for b in reinforce.dialog.buyers})
        rews.to_csv(os.path.join(self.args.savedir, f'res-{self.seed}.csv'))
        print('saved in', os.path.join(self.args.savedir, f'res-{self.seed}.csv'))

def main():
    parser = argparse.ArgumentParser(description='Reinforce')
    parser.add_argument('--alice_model_file', type=str,
        help='Alice model file')
    parser.add_argument('--bob_model_file', type=str,
        help='Bob model file')
    parser.add_argument('--joe_model_file', type=str,
        help='Joe model file')
    parser.add_argument('--output_model_file', type=str,
        help='output model file')
    parser.add_argument('--context_file', type=str,
        help='context file')
    parser.add_argument('--temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--pred_temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--verbose', action='store_true', default=False,
        help='print out conversations')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--score_threshold', type=int, default=6,
        help='successful dialog should have more than score_threshold in score')
    parser.add_argument('--log_file', type=str, default='',
        help='log successful dialogs to file for training')
    parser.add_argument('--smart_bob', action='store_true', default=False,
        help='make Bob smart again')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='discount factor')
    parser.add_argument('--eps', type=float, default=0.5,
        help='eps greedy')
    parser.add_argument('--momentum', type=float, default=0.1,
        help='momentum for sgd')
    parser.add_argument('--lr', type=float, default=0.1,
        help='learning rate')
    parser.add_argument('--clip', type=float, default=0.1,
        help='gradient clip')
    parser.add_argument('--rl_lr', type=float, default=0.002,
        help='RL learning rate')
    parser.add_argument('--rl_clip', type=float, default=2.0,
        help='RL gradient clip')
    parser.add_argument('--ref_text', type=str,
        help='file with the reference text')
    parser.add_argument('--sv_train_freq', type=int, default=-1,
        help='supervision train frequency')
    parser.add_argument('--nepoch', type=int, default=1,
        help='number of epochs')
    parser.add_argument('--hierarchical', action='store_true', default=False,
        help='use hierarchical training')
    parser.add_argument('--visual', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--domain', type=str, default='object_division',
        help='domain for the dialogue')
    parser.add_argument('--selection_model_file', type=str,  default='',
        help='path to save the final model')
    parser.add_argument('--data', type=str, default='data/negotiate',
        help='location of the data corpus')
    parser.add_argument('--unk_threshold', type=int, default=20,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--ctx_num', type=int, default=100,
        help='number of iterations')
    parser.add_argument('--delta', type=float, default=0.005,
        help='LOLA step sizes')
    parser.add_argument('--nu', type=float, default=1.0,
        help='LOLA step sizes')
    parser.add_argument('--engine_train', action='store_true', default=False,
        help='whether to train after each epoch')
    parser.add_argument('--validate', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--cooperate', action='store_true', default=False,
        help='whether to make the agents cooperate with themselves')
    parser.add_argument('--scratch', action='store_true', default=False,
        help='erase prediciton weights')
    parser.add_argument('--sep_sel', action='store_true', default=False,
        help='use separate classifiers for selection')
    parser.add_argument('--lola', action='store_true', default=False,
        help='use LOLA learning algorithm')
    parser.add_argument('--savedir', type=str, default='results/',
        help='path where to save results')

    args = parser.parse_args()

    utils.use_cuda(args.cuda)

    alice_model = utils.load_model(args.alice_model_file, cuda=args.cuda)
    alice = RlAgent(alice_model, args, name='Alice-Buyer', train=True)
    alice.vis = args.visual
    
    bob_model = utils.load_model(args.bob_model_file, cuda=args.cuda)
    bob = RlAgent(bob_model, args, name='Bob-Buyer', train=True)
    
    joe_model = utils.load_model(args.joe_model_file, cuda=args.cuda)
    joe = RlAgent(joe_model, args, name='Joe-Seller', train=True)

    dialog = TwoVsOneDialog([alice, bob], joe, args)
    logger = DialogLogger(verbose=args.verbose, log_file=args.log_file)
    ctx_gen = ContextGenerator(args.context_file, 3)

    domain = get_domain(args.domain)
    corpus = alice_model.corpus_ty(domain, args.data,
                                   freq_cutoff=args.unk_threshold,
                                   verbose=args.verbose, sep_sel=args.sep_sel)

    engine = alice_model.engine_ty(alice_model, args)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
        
    for seed in range(10):
        r = RLThread(args, seed, dialog, ctx_gen, engine, corpus, logger, tqdm)
        r.start()

    utils.save_model(alice.model, args.output_model_file)


if __name__ == '__main__':
    main()
