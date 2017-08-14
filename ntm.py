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

from tensorflow.python import debug as tf_debug

_NTMStateTuple = collections.namedtuple('NTMStateTuple', ('memory', 'controller', 'read', 'write'))

class NTMStateTuple(_NTMStateTuple):
  """Tuple used by NTM Cells for storing current state information
  """
  __slots__ = ()

def cosine_similarity(key, memory):
  return tf.matmul(tf.expand_dims(tf.nn.l2_normalize(key, dim=0), 0), tf.nn.l2_normalize(memory, dim=1), transpose_b=True)

def circular_convolution(weight, shift):
  """Performs a circular convolution on the weight vector. Currently hardcoded for shift of +/- 1 only
  """

  weight = tf.concat([weight[:,-1:], weight, weight[:,0:1]], axis=1)
  weight = tf.reshape(weight, [1, 1, int(weight.get_shape()[-1]), 1])
  shift = tf.reshape(shift, [1, int(shift.get_shape()[-1]), 1, 1])
  conv = tf.nn.conv2d(weight, shift, strides=[1,1,1,1], padding='VALID')
  return tf.reshape(conv, [1, int(conv.get_shape()[2])])


class FeedforwardController():
  """A wrapper class for a feedforward controller
  """

  def __init__(self, output_dim, activation=None):
    self.output_dim = output_dim
    self.activation = activation

  def __call__(self, _input, state):
    with tf.variable_scope('feedforward') as varscope:
      input_dim = _input.get_shape()[-1]
      w = tf.get_variable('w', shape=[input_dim, self.output_dim], dtype=tf.float32, initializer=tf.random_normal_initializer())
      b = tf.get_variable('b', shape=[self.output_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
      if self.activation is None:
        output = tf.nn.xw_plus_b(_input, w, b)
      else:
        output = self.activation(tf.nn.xw_plus_b(_input, w, b))
      return output, output

  def zero_state(self):
    return tf.zeros([1, self.output_dim])


class NTMCell():
  def __init__(self, controller, mem_dim=20, mem_size=128, read_heads=1, write_heads=1, shift_dim=1, output_dim=10):
    self.mem_dim = mem_dim
    self.mem_size = mem_size
    self.controller = controller
    self.controller_dim = controller.output_dim
    self.num_read_heads = read_heads
    self.num_write_heads = write_heads
    self.shift_dim = shift_dim
    self.output_dim = output_dim

  def __call__(self, _inputs, state):
    with tf.variable_scope('ntm_cell'):
      (m_prev, ctrl_prev, r_prev, w_prev_weights) = state

      r_prev_weights, r_prev = tuple(zip(*r_prev))

      with tf.variable_scope('controller'):
        ctrl_output, ctrl_state = self.build_controller(_inputs, r_prev, ctrl_prev)
      with tf.variable_scope('memory'):
        read = []
        erases = []
        adds = []
        write = []
        for i, prev_weight in enumerate(r_prev_weights):
          with tf.variable_scope('read_head_%d' %i):
            read.append(self.read_head(m_prev, ctrl_output, prev_weight))

        for i, prev_weight in enumerate(w_prev_weights):
          with tf.variable_scope('write_head_%d' %i):
            w, e, a = self.write_head(m_prev, ctrl_output, prev_weight)
            write.append(w)
            erases.append(e)
            adds.append(a)

        mem = m_prev
        for e in erases:
          mem = mem * e

        mem = mem + tf.add_n(adds)

      state = NTMStateTuple(mem, ctrl_state, read, write)

      with tf.variable_scope('output'):
        w = tf.get_variable('w', shape=[self.controller_dim, self.output_dim], dtype=tf.float32, initializer=tf.random_normal_initializer())
        b = tf.get_variable('b', shape=[self.output_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
        ctrl_output = tf.nn.xw_plus_b(ctrl_output, w, b)
      return ctrl_output, state

  def zero_state(self):

    mem = tf.truncated_normal([self.mem_size, self.mem_dim], mean=0.5, stddev=0.2)
    read = []
    write = []
    for i in range(self.num_read_heads):
      read.append((tf.fill([1, self.mem_size], 1e-6), tf.fill([1, self.mem_dim], 1e-6)))
    for i in range(self.num_write_heads):
      write.append(tf.fill([1, self.mem_size], 1e-6))
    return NTMStateTuple(mem, self.controller.zero_state(), read, write)

  
  def build_controller(self, _inputs, r_prev, ctrl_state):
    ctrl_input = tf.concat([_inputs] + list(r_prev), axis=1)
    return self.controller(ctrl_input, ctrl_state)
  
  def get_weight(self, memory, hidden_state, prev_w):
    output_dim = self.mem_dim + 3 + self.shift_dim * 2 + 1
    with tf.variable_scope('addressing') as varscope:
      w = tf.get_variable('w', shape=[self.controller_dim, output_dim], dtype=tf.float32)
      b = tf.get_variable('b', shape=[output_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
      out = tf.squeeze(tf.nn.xw_plus_b(hidden_state, w, b))

      key = tf.slice(out, begin=[0], size=[self.mem_dim], name='key')
      beta = tf.nn.softplus(out[self.mem_dim:self.mem_dim+1], name='beta')
      gate = tf.nn.sigmoid(out[self.mem_dim+1:self.mem_dim+2], name='gate')
      gamma = tf.nn.softplus(out[self.mem_dim+2:self.mem_dim+3], name='gamma')
      shift = tf.nn.softmax(out[self.mem_dim+3:], name='shift')

      content_w = tf.nn.softmax(beta*cosine_similarity(key, memory))
      gated_w = gate*content_w + (1.0 - gate)*prev_w
      weight = circular_convolution(gated_w, shift)
      powed_weight = tf.pow(weight, gamma)
      return powed_weight/(tf.reduce_sum(powed_weight) + 1e-12)

  def read_head(self, memory, hidden_state, prev_w):
    weight = self.get_weight(memory, hidden_state, prev_w)
    read = tf.matmul(weight, memory)
    return weight, read

  def write_head(self, memory, hidden_state, prev_w):
    weight = self.get_weight(memory, hidden_state, prev_w)
    with tf.variable_scope('erase') as varscope:
      w = tf.get_variable('w', shape=[self.controller_dim, self.mem_dim], dtype=tf.float32, initializer=tf.random_normal_initializer())
      b = tf.get_variable('b', shape=[self.mem_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
      erase = tf.nn.sigmoid(tf.nn.xw_plus_b(hidden_state, w, b))
      erase = tf.ones([self.mem_size, self.mem_dim], dtype=tf.float32) - tf.matmul(weight, erase, transpose_a=True)
    with tf.variable_scope('add') as varscope:
      w = tf.get_variable('w', shape=[self.controller_dim, self.mem_dim], dtype=tf.float32, initializer=tf.random_normal_initializer())
      b = tf.get_variable('b', shape=[self.mem_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
      add = tf.nn.tanh(tf.nn.xw_plus_b(hidden_state, w, b))
      add = tf.matmul(weight, add, transpose_a=True)
    return weight, erase, add


def dynamic_ntm(cell, _inputs):
  """Dynamic unrolling of an NTM Cell. Similar to the tf.nn.dynamic_rnn function.

  Args:
    cell: an NTM cell
    _inputs: A 'Tensor' of shape [time, batch_size, input_size]

  Returns:
    A tuple (outputs, read_ws, write_ws, mem_states, final_state) where:

      outputs: The controller hidden states at each time step

      read_ws: A list containing the read weights from each read head at each time step

      write_ws: A list containing the write weights from each write head at each time step

      mem_states: A tensor containing the memory contents at each time step

      final_state: The final state tuple

  """

  with tf.variable_scope('ntm') as varscope:
    time_steps = tf.shape(_inputs)[0]

    state = cell.zero_state()

    with tf.name_scope('dynamic_ntm') as scope:
      base_name = scope

    def _create_ta(name, dtype):
      return tf.TensorArray(dtype=dtype, size=time_steps, tensor_array_name=base_name + name)

    input_ta = _create_ta('input', _inputs.dtype)
    input_ta = input_ta.unstack(_inputs)
    output_ta = _create_ta('output', _inputs.dtype)
    read_w_ta = _create_ta('read_w', _inputs.dtype)
    write_w_ta = _create_ta('write_w', _inputs.dtype)
    mem_state_ta = _create_ta('mem_state', _inputs.dtype)

    time = tf.constant(0, dtype=tf.int32, name='time')

    def _time_step(time, output_ta_t, read_w_ta_t, write_w_ta_t, mem_state_ta_t, state):
      input_t = input_ta.read(time)
      call_cell = lambda: cell(input_t, state)
      (output, new_state) = call_cell()

      mem_state, ctrl_state, read_states, write_weights = new_state
      read_weights, read_contents = tuple(zip(*read_states))

      read_w_ta_t = read_w_ta_t.write(time, read_weights[0])
      write_w_ta_t = write_w_ta_t.write(time, write_weights[0])
      mem_state_ta_t = mem_state_ta_t.write(time, mem_state)

      output_ta_t = output_ta_t.write(time, output)
      return (time + 1, output_ta_t, read_w_ta_t, write_w_ta_t, mem_state_ta_t, new_state)

    _, output_final_ta, read_final_ta, write_final_ta, mem_state_final_ta, final_state = tf.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_time_step,
        loop_vars=(time, output_ta, read_w_ta, write_w_ta, mem_state_ta, state),
        parallel_iterations=32,
        swap_memory=True)

    outputs = output_final_ta.stack()
    read_ws = read_final_ta.stack()
    write_ws = write_final_ta.stack()
    mem_states = mem_state_final_ta.stack()

    return outputs, read_ws, write_ws, mem_states, final_state


class NTMModel():
  def __init__(self, ntm_cell, train=False, batch_size=1, input_dim=10, target_dim=10):
    self.input_dim = input_dim
    self.target_dim = target_dim
    self.learning_rate = 1e-4
    self.momentum = 0.9
    self.global_step = 0

    with tf.variable_scope('inputs'):
      self.inputs = tf.placeholder(tf.float32, shape=[None, batch_size, self.input_dim])
      self.targets = tf.placeholder(tf.float32, shape=[None, batch_size, self.target_dim])

    with tf.variable_scope('inference'):
      self.ntm_cell = ntm_cell
      self.final_outputs, self.final_reads, self.final_writes, self.final_mem_states, self.final_state = dynamic_ntm(ntm_cell, self.inputs)
      self.predictions = tf.clip_by_value(tf.nn.sigmoid(self.final_outputs), 1e-6, 1. - 1e-6)

    if train:
      with tf.variable_scope('train'):
        self.loss = tf.reduce_mean(-1 * self.targets * tf.log(self.predictions) - (1 - self.targets) * tf.log(1 - self.predictions))
        tf.summary.scalar('loss', self.loss)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum)
        gradients = optimizer.compute_gradients(self.loss)

        for i, (grad, var) in enumerate(gradients):
          if grad is not None:
            gradients[i] = (tf.clip_by_value(grad, -10, 10), var)
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/grad', grad)

        self.train_op = optimizer.apply_gradients(gradients)

    self.init_op = tf.global_variables_initializer()
    self.saver = tf.train.Saver()

  def add_summary(self, logdir, graph):
    self.summary_op = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(logdir, graph=graph)

  def save(self, session, ckpt_dir):
    self.saver.save(session, os.path.join(ckptdir, 'model-ckpt'), global_step=self.global_step)

  def restore(self, session, ckpt_dir):
    self.saver.restore(session, tf.train.latest_checkpoint(ckpt_dir))
    print('Restored model from checkpoint')

  def initialize_model(self, session):
    session.run(self.init_op)

  def train_step(self, session, _input, target, summarize=False):
    feed_dict = {
      self.inputs: _input,
      self.targets: target
    }

    if summarize:
      predictions, read_weights, write_weights, memory, summary, _ = session.run([self.predictions, 
                                                                                  self.final_reads, 
                                                                                  self.final_writes, 
                                                                                  self.final_mem_states, 
                                                                                  self.summary_op,
                                                                                  self.train_op],
                                                                                  feed_dict=feed_dict)

      self.summary_writer.add_summary(summary, self.global_step)

      plot_data = {
        'input': np.squeeze(_input),
        'output': np.round(np.squeeze(predictions)),
        'read_weights': np.squeeze(read_weights),
        'write_weights': np.squeeze(write_weights),
        'memory': memory
      }

      self.global_step += 1
      return plot_data

    else:
      session.run(self.train_op, feed_dict=feed_dict)
      self.global_step += 1
      return None

  def inference_step(self, session, _input, target):
    feed_dict = {
      self.inputs: _input,
      self.targets: target
    }

    predictions, read_weights, write_weights, memory = session.run([self.predictions,
                                                                   self.final_reads,
                                                                   self.final_writes,
                                                                   self.final_mem_states],
                                                                   feed_dict=feed_dict)
    plot_data = {
      'input': np.squeeze(_input),
      'output': np.round(np.squeeze(predictions)),
      'read_weights': np.squeeze(read_weights),
      'write_weights': np.squeeze(write_weights),
      'memory': memory
    }

    return plot_data


