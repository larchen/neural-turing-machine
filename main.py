import os
import sys
import argparse
import collections
import json
import re
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

from ntm import *
from utils import *

FLAGS = None

def create_copy_example(length):
  """Creates a single copy example

  Args:
    length: the length of the copy sequence.

  Returns:
    A tuple (ex_input, ex_target), both with shape [2*length+2, batch_size, 10]

  """
  start = np.zeros((1, 1, 10))
  end = np.zeros((1, 1, 10))
  start[0][0][0] = 1
  end[0][0][1] = 1
  seq = generate_copy_sequence(length, 8)
  ex_input = np.concatenate((start, seq, end, np.zeros((length, 1, 10))))
  ex_target = np.concatenate((np.zeros((length + 2, 1, 10)), seq))
  return ex_input, ex_target

def run_training(num_epochs):
  logdir = create_rundir('logs')
  ckptdir = create_rundir('checkpoints')
  
  with tf.Session() as sess:
    controller = FeedforwardController(output_dim=100)
    ntm_cell = NTMCell(controller)
    ntm_model = NTMModel(ntm_cell, train=True)

    ntm_model.add_summary(logdir, sess.graph)
    
    ntm_model.initialize_model(sess)
    fig, axes = initialize_plots()

    for step in range(num_epochs):
      max_length = 2 + int(step*19.0/100000)

      random_length = np.random.randint(1, 20+1)
      _input, target = create_copy_example(random_length)

      print('\rStep: %d' %step, end='')

      if step%100 == 0:
        plot_data = ntm_model.train_step(sess, _input, target, summarize=True)
        plot_copy_sequence(fig, axes, plot_data, step)
      else:
        ntm_model.train_step(sess, _input, target)

def run_inference(ckptdir):

  with tf.Session() as sess:
    controller = FeedforwardController(output_dim=100)
    ntm_cell = NTMCell(controller)
    ntm_model = NTMModel(ntm_cell, train=False)

    ntm_model.initialize_model(sess)
    fig, axes = initialize_plots()
    
    ntm_model.restore(sess, ckptdir)

    while(True):
      copy_length = int(input('Sequence length: '))
      _input, target = create_copy_example(copy_length)

      plot_data = ntm_model.inference_step(sess, _input, target)

      plot_copy_sequence(fig, axes, plot_data, 0)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', action='store_true', dest='is_training', 
                      help='Train a new model')
  parser.add_argument('--iterations', type=int, default='100000', 
                      help='Number of training epochs')
  parser.add_argument('--checkpoint', type=str, default='checkpoints/pretrained', 
                      help='Checkpoint directory to restore model from')
  FLAGS, unparsed = parser.parse_known_args()
  
  if FLAGS.is_training:
    run_training(FLAGS.iterations)
  else:
    run_inference(FLAGS.checkpoint)