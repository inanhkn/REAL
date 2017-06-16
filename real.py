# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is based on an early version of the tensorflow example on 
# language modeling with PTB (at /models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py)
#
# Hakan Inan & Khashayar Khosravi
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import shutil
import numpy as np
import tensorflow as tf
from copy import deepcopy

import reader
from reader import _build_vocab


def get_model_name(config):
  return "keep_prob=%s"%config.keep_prob + \
      "_size=%s"%config.hidden_size + \
      "_AL_weight=%s"%config.weight_AL + \
      "_temp=%s"%config.temperature + \
      "_batch_size=%s"%config.batch_size+ \
      "_num_steps=%s"%config.num_steps + \
      "_max_epoch=%s"%config.max_epoch + \
      "_lr_decay=%s"%config.lr_decay + \
      "_reuse_embed=%s"%config.reuse_embedding

class REAL(object):
  """2-Layer LSTM language model with RE and AL options"""

  def __init__(self, is_training, config):
    self.model_name = get_model_name(config)

    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])


    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0) 
    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)


    self._initial_state = tf.concat(1,
      [lstm_cell_1.zero_state(batch_size, tf.float32),
      lstm_cell_2.zero_state(batch_size, tf.float32)])


    embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)

    self.embeddings = embedding

    inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    outputs = []
    state = self._initial_state

    if not config.reuse_embedding:
      softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [1,vocab_size])    

    # Decompose into states for each LSTM layer
    state_1,state_2 = tf.split(1,2,state)

    '''
    Dropout masks. Here we use a different variational dropout
    than described in (Gal & Gahramani, 2016). Specifically, we
    use the same masks for LSTM states when used as hidden
    states and as outputs.
    '''
    C_input = tf.ones([batch_size,size],tf.float32)
    C_state = tf.ones([batch_size,size*2],tf.float32)
    dM_input = tf.nn.dropout(C_input,config.keep_prob)
    dM_state_1 = tf.nn.dropout(C_state,config.keep_prob)
    dM_state_1 = tf.concat(1,[C_state[:,:size],dM_state_1[:,size:]])
    dM_state_2 = tf.nn.dropout(C_state,config.keep_prob)
    dM_state_2 = tf.concat(1,[C_state[:,:size],dM_state_2[:,size:]])
    _,dM_out_1 = tf.split(1,2,dM_state_1)
    _,dM_out_2 = tf.split(1,2,dM_state_2)
    

    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()

        # Dropout for input:
        current_input = inputs[:, time_step, :]
        if is_training and config.keep_prob < 1:
          current_input  = tf.mul(dM_input,current_input)

        # Run LSTM1:
        with tf.variable_scope("LSTM1"):
          (out_1,state_1) = lstm_cell_1(current_input,state_1)

        # Dropout for output from LSTM1 to LSTM2:
        if is_training and config.keep_prob < 1:
          out_1 = tf.mul(dM_out_1,out_1)

        # Dropout for state_1:
        if is_training and config.keep_prob < 1:
          state_1 = tf.mul(dM_state_1,state_1)         

        # Run LSTM2:
        with tf.variable_scope("LSTM2"):
          (out_2,state_2) = lstm_cell_2(out_1,state_2)

        # Dropout for final output
        if is_training and config.keep_prob < 1:
          out_2 = tf.mul(dM_out_2,out_2)

        # Dropout for state_2:
        if is_training and config.keep_prob < 1:
          state_2 = tf.mul(dM_state_2,state_2)

        outputs.append(out_2)

      # Compose the final state from those of individual layers
      state = tf.concat(1,[state_1,state_2])
    

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    self.L = embedding
    if config.reuse_embedding:
      logits = tf.matmul(output,tf.transpose(embedding))
    else:
      logits = tf.matmul(output,softmax_w)+softmax_b
      self.W = softmax_w
      self.b = softmax_b

    labels = tf.reshape(self._targets, [-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels)

    if config.weight_AL>0:
      output_vectors = tf.gather(embedding,labels)
      soft_logits= tf.matmul(output_vectors,tf.transpose(embedding))
      soft_labels = tf.nn.softmax(soft_logits/config.temperature)
      AL = tf.nn.softmax_cross_entropy_with_logits(logits/config.temperature,soft_labels) # Cross entropy
      AL += tf.reduce_sum(tf.log(soft_labels)*soft_labels,reduction_indices=1) # -entropy
      self._cost = cost = tf.reduce_sum(config.weight_AL*AL+loss) / batch_size

    else:
      self._cost = cost = tf.reduce_sum(loss) / batch_size

    self._lr = tf.Variable(0.0, trainable=False)
    self._perp = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return    

    
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.optimizer = optimizer
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def perp(self):
    return self._perp

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

class PTB_large_config(object):
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 6
  num_steps = 35
  hidden_size = 1500
  max_epoch = 1
  max_max_epoch = 100
  keep_prob = 0.35
  lr_decay = 0.97
  batch_size = 20
  temperature = 20
  weight_AL = 0.0
  early_stopping = 5
  reuse_embedding = True
  save_models=False
  use_saved=False


def run_epoch(session, m, data, eval_op,save_models=False,verbose=False):
  saver = tf.train.Saver()
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  perps = 0.0
  iters = 0
  state = m.initial_state.eval()

  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):

    perp, cost, state, _ = session.run([m.perp,m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})                                 

    costs += cost
    perps += perp
    iters += m.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%d / %d exp_cost: %.3f perplexity: %.3f speed: %.0f wps" %
            (step, epoch_size,np.exp(costs / iters), np.exp(perps / iters),
             iters * m.batch_size / (time.time() - start_time)))
  if save_models:
  	saver.save(session, 'saved_models/%s.temp'%m.model_name,write_meta_graph=False)
  return np.exp(perps / iters)


def run_REAL(hidden_size=None,keep_prob=None,weight_AL=None,temperature=None,init_scale=None,
  batch_size=None,num_steps=None,max_epoch=None,lr_decay=None,max_grad_norm=None,
  reuse_embedding=None,max_max_epoch=None,save_models=None,use_saved=None,data_path = "data"):
  '''
  Runs the model with AL and RE options.

    inputs:
  		hidden_size: # of units for input & hidden layers
  		keep_prob: Probability of not zeroing a unit for dropout
  		weight_AL: Weight used for the eaugmented loss (beta in the paper)
  		temperature: Temperature used for the y terms in the augmented loss
  		init_scale: Scale for rendom initialization of weights
  		batch_size: Size of each minibatch
  		num_steps: Number of unrolled time steps
  		max_epoch: last epoch before learning rate starts to decay
  		lr_decay: decay rate for the learning rate of SGD
  		max_grad_norm: maximum gradient norm after clipping by the global norm
  		reuse_embedding: embedding and output matrices are tied if True
  		max_max_epoch: index of the final epoch
  		save_model: weights are saved in '/saved_models' if True
  		use_saved: Model is initialized with saved weights from same config if True
  		data_path: path to dataset - Files must be named 'train.txt','valid.txt','test.txt'

  	returns: 
  		train_perplexity, best_valid_perplexity, test_perplexity


  '''
  raw_data = reader.ptb_raw_data(data_path)
  train_data, valid_data, test_data, vocab_size = raw_data

  config = PTB_large_config()
  config.vocab_size = vocab_size

  if hidden_size is not None:
    config.hidden_size = hidden_size
  if keep_prob is not None:
    config.keep_prob = keep_prob
  if temperature is not None:
    config.temperature = temperature
  if batch_size is not None:
    config.batch_size = batch_size
  if num_steps is not None:
    config.num_steps = num_steps
  if max_epoch is not None:
    config.max_epoch = max_epoch
  if max_max_epoch is not None:
    config.max_max_epoch = max_max_epoch
  if lr_decay is not None:
    config.lr_decay = lr_decay
  if init_scale is not None:
    config.init_scale = init_scale
  if max_grad_norm is not None:
    config.max_grad_norm = max_grad_norm
  if weight_AL is not None:
    config.weight_AL = weight_AL
  if reuse_embedding is not None:
    config.reuse_embedding = reuse_embedding
  if save_models is not None:
  	config.save_models = save_models
  if use_saved is not None:
  	config.use_saved = use_saved

  if config.save_models:
  	if not os.path.exists('saved_models'):
  		os.mkdir('saved_models')


  test_config = deepcopy(config)
  test_config.batch_size = 1
  test_config.num_steps = 1
  test_config.keep_prob =1.0

  best_valid_perplexity = 1e10
  best_val_epoch = 0

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = REAL(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = REAL(is_training=False, config=config)
      mtest = REAL(is_training=False, config=test_config)

    if config.use_saved and os.path.isfile('saved_models/%s'%m.model_name):
      saver = tf.train.Saver()
      saver.restore(session,'saved_models/%s'%m.model_name)
    else:
      tf.initialize_all_variables().run()

    
    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0)
      m.assign_lr(session, config.learning_rate * lr_decay)    

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr),))
      train_perplexity = run_epoch(session, m, train_data, m.train_op,
      	save_models=config.save_models, verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      if valid_perplexity < best_valid_perplexity:
      	if config.save_models:
          shutil.copyfile('saved_models/%s.temp'%m.model_name, 'saved_models/%s'%m.model_name)
        best_valid_perplexity = valid_perplexity
        best_val_epoch = i
      if i - best_val_epoch > config.early_stopping:
        print("Early stop in effect (Epoch %d)" %(i) )
        break

    
    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)
    return train_perplexity,best_valid_perplexity,test_perplexity


if __name__ == "__main__":
  run_REAL()