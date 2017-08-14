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

FLAGS = None

def generate_copy_sequence(length, bits):
  seq = np.zeros([length, 1, bits + 2], dtype=np.float32)
  for idx in range(length):
    seq[idx, 0, 2:bits+2] = np.random.rand(bits).round()
  return seq

def create_rundir(_dir):
    _dir = os.path.expanduser(_dir)
    if not os.path.isdir(_dir):
        print('directory \'%s\' does not exist' %_dir)
        return ''

    runs = [int(r) for r in os.listdir(_dir) if r.isdigit()]
    new_run = 1 if not runs else max(runs) + 1
    os.mkdir(os.path.join(_dir, str(new_run)))
    return os.path.join(_dir, str(new_run))

def initialize_plots():
  plt.ion()
  plt.show()

  fig = plt.figure(figsize=(10,10))
  ax0 = plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=1)
  ax1 = plt.subplot2grid((3,4), (1,0), colspan=2, rowspan=1)
  ax2 = plt.subplot2grid((3,4), (0,2), colspan=1, rowspan=3)
  ax3 = plt.subplot2grid((3,4), (0,3), colspan=1, rowspan=3)
  ax4 = plt.subplot2grid((3,4), (2,0), colspan=2, rowspan=1)

  axes = [ax0, ax1, ax2, ax3, ax4]

  plt.tight_layout()
  return fig, axes

def plot_copy_sequence(fig, axes, plot_data, step):
  [ax.cla() for ax in axes]

  axes[0].set_title('Input')
  axes[1].set_title('Output')
  axes[2].set_title('Read Weights')
  axes[3].set_title('Write Weights')
  axes[4].set_title('Memory')

  axes[0].imshow(np.transpose(plot_data['input']), vmin=0, vmax=1, cmap='gray')
  axes[0].set_xlabel('Time')

  axes[1].imshow(np.transpose(plot_data['output']), vmin=0, vmax=1, cmap='gray')
  axes[1].set_xlabel('Time')

  axes[2].imshow(np.transpose(plot_data['read_weights']), vmin=0, vmax=1, cmap='gray')
  axes[2].set_xlabel('Time')
  axes[2].set_ylabel('Memory Location')

  axes[3].imshow(np.transpose(plot_data['write_weights']), vmin=0, vmax=1, cmap='gray')
  axes[3].set_xlabel('Time')
  axes[3].set_ylabel('Memory Location')

  ims = [[axes[4].imshow(np.transpose(mem), cmap='gray', animated=True, vmin=0, vmax=1), axes[4].text(2, -2, 't = %d' %i, animated=True)] for i, mem in enumerate(plot_data['memory'])]
  ani = animation.ArtistAnimation(fig, ims, interval=400, blit=False, repeat_delay=20)

  plt.tight_layout()
  plt.draw()
  plt.pause(0.001)

def plot_memory(mem_states):
  fig = plt.figure()

  # ims = [[plt.imshow(mem, cmap='gray', animated=True)] for mem in mem_states]
  ims = []
  for mem in mem_states:
    print(mem)
    im = plt.imshow(mem, vmin=0, vmax=1, cmap='gray', animated=True)
    ims.append([im])

  ani = animation.ArtistAnimation(fig, ims, interval=500, blit=False, repeat_delay=1000)
  plt.show()

