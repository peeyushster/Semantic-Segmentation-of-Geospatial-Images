#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 3 00:20:42 2018

@author: serkankarakulak
"""
import argparse
import multiprocessing as mp
from model import *


parser = argparse.ArgumentParser(description='Trains a model to recover the true signal, given noisy observations with cyclic shifts')
parser.add_argument('-N','--train-n-batches', default=1, help='Number of batches to train with',required=True)
parser.add_argument('-l','--learning-rate', default=0.001, help='Initial learning rate')
parser.add_argument('-b','--minibatch-size', default=16, help='Minibatch size during the training')
parser.add_argument('-t','--test-sample-size', default=4, help='Sample size of the test batches')
parser.add_argument('-e','--eval-n-times', default=4, help='Number of batches to be tested at evaluation')
parser.add_argument('-v','--version', default='v1', help='Version name of the model')
parser.add_argument('-p','--num-of-processes', default=7, help="Number of parallel processes to be used for data preparation (999999 for 'maxthreadnum - 1')")
parser.add_argument('-s','--test-split', default=0.2, help='Percentage of tiles reserved for test data')
parser.add_argument('-c','--n-class', default=3, help='Number of classes')
parser.add_argument('-i','--img-size', default=256, help='Image size')
parser.add_argument('-k','--skip-step', default=100, help='Frequency of test loss logs')
parser.add_argument('-a','--eval-after', default=0, help='After which step to start generating test loss logs')

args = parser.parse_args()

threadNum = mp.cpu_count()-1 if args.num_of_processes == 999999 else args.num_of_processes


model = U_Net(
   lr=float(args.learning_rate),
   batchSize=int(args.minibatch_size),
   validBatchSize=int(args.test_sample_size),
   evalNTimes=int(args.eval_n_times),
   vers=args.version,
   nProcessesDataPrep=int(args.num_of_processes),
   testSplit = float(args.test_split),
   numClasses=int(args.n_class),
   imgSize=int(args.img_size),
   skipStep=int(args.skip_step),
   evalAfterStep=int(args.eval_after),
   isTraining = True
   )

model.build()

print('\nversion name: ' + model.vers +'\n')

model.train(nBatches=int(args.train_n_batches))
