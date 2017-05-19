#! /usr/bin/env python
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script to run training and evaluation of models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile

import yaml
import json

import tensorflow as tf
import numpy as np
import pdb
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from seq2seq.extra.text_cnn import TextCNN
from seq2seq.data import vocab, input_pipeline
from seq2seq.models.attention_seq2seq import AttentionSeq2Seq
from seq2seq.test import utils as test_utils
from seq2seq.encoders.rnn_encoder import BidirectionalRNNEncoder
from tensorflow import gfile

from seq2seq import models
from seq2seq.contrib.experiment import Experiment as PatchedExperiment
from seq2seq.configurable import _maybe_load_yaml, _create_from_dict
from seq2seq.configurable import _deep_merge_dict
from seq2seq.data import input_pipeline
from seq2seq.metrics import metric_specs
from seq2seq.training import hooks
from seq2seq.training import utils as training_utils


tf.flags.DEFINE_string("config_paths", "",
                       """Path to a YAML configuration files defining FLAG
                       values. Multiple files can be separated by commas.
                       Files are merged recursively. Setting a key in these
                       files is equivalent to setting the FLAG value with
                       the same name.""")
tf.flags.DEFINE_string("hooks", "[]",
                       """YAML configuration string for the
                       training hooks to use.""")
tf.flags.DEFINE_string("metrics", "[]",
                       """YAML configuration string for the
                       training metrics to use.""")
tf.flags.DEFINE_string("model", "",
                       """Name of the model class.
                       Can be either a fully-qualified name, or the name
                       of a class defined in `seq2seq.models`.""")
tf.flags.DEFINE_string("model_params", "{}",
                       """YAML configuration string for the model
                       parameters.""")

tf.flags.DEFINE_string("input_pipeline_train", "{}",
                       """YAML configuration string for the training
                       data input pipeline.""")
tf.flags.DEFINE_string("input_pipeline_dev", "{}",
                       """YAML configuration string for the development
                       data input pipeline.""")

tf.flags.DEFINE_string("buckets", None,
                       """Buckets input sequences according to these length.
                       A comma-separated list of sequence length buckets, e.g.
                       "10,20,30" would result in 4 buckets:
                       <10, 10-20, 20-30, >30. None disabled bucketing. """)
tf.flags.DEFINE_integer("batch_size", 16,
                        """Batch size used for training and evaluation.""")
tf.flags.DEFINE_string("output_dir", None,
                       """The directory to write model checkpoints and summaries
                       to. If None, a local temporary directory is created.""")

# Training parameters
tf.flags.DEFINE_string("schedule", "continuous_train_and_eval",
                       """Estimator function to call, defaults to
                       continuous_train_and_eval for local run""")
tf.flags.DEFINE_integer("train_steps", None,
                        """Maximum number of training steps to run.
                         If None, train forever.""")
tf.flags.DEFINE_integer("eval_every_n_steps", 1000,
                        "Run evaluation on validation data every N steps.")

# RunConfig Flags
tf.flags.DEFINE_integer("tf_random_seed", None,
                        """Random seed for TensorFlow initializers. Setting
                        this value allows consistency between reruns.""")
tf.flags.DEFINE_integer("save_checkpoints_secs", None,
                        """Save checkpoints every this many seconds.
                        Can not be specified with save_checkpoints_steps.""")
tf.flags.DEFINE_integer("save_checkpoints_steps", None,
                        """Save checkpoints every this many steps.
                        Can not be specified with save_checkpoints_secs.""")
tf.flags.DEFINE_integer("keep_checkpoint_max", 5,
                        """Maximum number of recent checkpoint files to keep.
                        As new files are created, older files are deleted.
                        If None or 0, all checkpoint files are kept.""")
tf.flags.DEFINE_integer("keep_checkpoint_every_n_hours", 4,
                        """In addition to keeping the most recent checkpoint
                        files, keep one checkpoint file for every N hours of
                        training.""")
tf.flags.DEFINE_float("gpu_memory_fraction", 1.0,
                      """Fraction of GPU memory used by the process on
                      each GPU uniformly on the same machine.""")
tf.flags.DEFINE_boolean("gpu_allow_growth", False,
                        """Allow GPU memory allocation to grow
                        dynamically.""")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        """Log the op placement to devices""")


FLAGS = tf.flags.FLAGS

def translate(sentence):
    with open('../autoencoder/processed_data/vocab.json','r') as fp:
        word_dict = json.load(fp)

    sentence_str = []
    for id in list(sentence):
        word = word_dict.get(str(id))
        sentence_str.append(word)
#        print word,id
    print (" ".join(sentence_str))

def get_vars_from_scope(scope):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
    names = list(v.name for v in vars)
    return vars, names

def pad_sentence(input_tensor):
    reconstructed_shape = tf.shape(input_tensor)
    padded_shape=[128,73 - reconstructed_shape[1]]
    padded_vals = tf.fill(padded_shape,35880)
    return tf.concat([input_tensor,padded_vals],
                                      axis=1)

def create_model(mode, params=None):
    params_ = AttentionSeq2Seq.default_params().copy()
    params_.update(params or {})
    return AttentionSeq2Seq(params=params_, mode=mode)

def create_experiment(output_dir):
  """
  Creates a new Experiment instance.

  Args:
    output_dir: Output directory for model checkpoints and summaries.
  """

  config = run_config.RunConfig(
      tf_random_seed=FLAGS.tf_random_seed,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      gpu_memory_fraction=FLAGS.gpu_memory_fraction)
  config.tf_config.gpu_options.allow_growth = FLAGS.gpu_allow_growth
  config.tf_config.log_device_placement = FLAGS.log_device_placement

  train_options = training_utils.TrainOptions(
      model_class=FLAGS.model,
      model_params=FLAGS.model_params)
  # On the main worker, save training options
  if config.is_chief:
    gfile.MakeDirs(output_dir)
    train_options.dump(output_dir)

  bucket_boundaries = None
  if FLAGS.buckets:
    bucket_boundaries = list(map(int, FLAGS.buckets.split(",")))

  # Training data input pipeline
  train_input_pipeline = input_pipeline.make_input_pipeline_from_def(
      def_dict=FLAGS.input_pipeline_train,
      mode=tf.contrib.learn.ModeKeys.TRAIN)

  # Create training input function
  train_input_fn = training_utils.create_input_fn(
      pipeline=train_input_pipeline,
      batch_size=FLAGS.batch_size,
      bucket_boundaries=bucket_boundaries,
      scope="train_input_fn")

  model = create_model(mode=tf.contrib.learn.ModeKeys.TRAIN,
          params=train_options.model_params)

  data_provider = train_input_pipeline.make_data_provider()

  features, labels = train_input_fn()

  filter_sizes="3,4,5"
  cnn = TextCNN(sequence_length=73,
            num_classes=2,
            vocab_size=35883,
            embedding_size=128,
            filter_sizes=list(map(int,filter_sizes.split(","))),
            num_filters=128,
            l2_reg_lambda=0.0) 

  cnn1 = TextCNN(sequence_length=73,
            num_classes=2,
            vocab_size=35883,
            embedding_size=128,
            filter_sizes=list(map(int,filter_sizes.split(","))),
            num_filters=128,
            l2_reg_lambda=0.0,
            name_scope='classifierCNN') 


  fetches = model(features,labels,None)
  predictions_, loss_, train_op_ = fetches

  reconstructed_sentences = pad_sentence(predictions_['predicted_ids'])
  input_sentences = pad_sentence(tf.cast(predictions_['features.source_ids'],
                                                             dtype=tf.int32))
  input_y = tf.concat([tf.zeros([128,1]),tf.ones([128,1])],axis=1)

  real_scores, real_labels = cnn.inference(reconstructed_sentences)
  real_acc = cnn.accuracy(real_labels,input_y)
#  tf.get_variable_scope().reuse_variables()
  fake_scores, fake_labels = cnn1.inference(input_sentences)
  fake_acc = cnn.accuracy(fake_labels,input_y)

#  pdb.set_trace()
  losses = tf.nn.softmax_cross_entropy_with_logits( \
          logits=tf.cast(tf.argmax(fake_scores,axis=1),dtype=tf.float64), \
          labels=tf.cast(real_labels,dtype=tf.float64))

  distance_loss = tf.multiply(tf.reduce_mean(losses),1.0)

#  pdb.set_trace()
  total_loss = tf.cast(distance_loss,tf.float32) + loss_

  train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(total_loss)

  #######################Add another loss term#####################
  #_, real_labels = infer_classifier(cnn,session,
  #        batch_data_padded,classifier_dir)

  #reconstructed_sentences = session.run(model.decoder_prediction_train, fd)
  #reconstructed_sentences_padded = np.zeros(batch_data_padded.shape)
  #reconstructed_sentences_padded[:reconstructed_sentences.T.shape[0],
  #        :reconstructed_sentences.T.shape[1]]
  #reconstructed_scores, fake_labels = infer_classifier(cnn,
  #        session, reconstructed_sentences_padded, classifier_dir)

  #losses = tf.nn.softmax_cross_entropy_with_logits(logits=reconstructed_scores,
  #        labels = real_labels)
  #distance_loss = tf.multiply(tf.reduce_mean(losses),10)
  #loss = model.loss + distance_loss

  #### Loss summaries

  #train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
  ################################################################

  ae_vars, _ = get_vars_from_scope(scope='model')

  ae_saver = tf.train.Saver(var_list = ae_vars)
  ae_saved_model = tf.train.get_checkpoint_state(output_dir)

  classifier_dir = '../../text_autoencoder/autoencoder/runs/1495150345/checkpoints'
  classifier_dir_1 = '../../text_autoencoder/autoencoder/runs/1495166749/checkpoints'
  classifier_vars, _ = get_vars_from_scope(scope='TextCNN')
  classifier_vars_1, _ = get_vars_from_scope(scope='classifierCNN')
  print ([v.name for v in classifier_vars])
  print ([v.name for v in classifier_vars_1])

  classifier_saver = tf.train.Saver(var_list = classifier_vars)
  classifier_saved_model = tf.train.get_checkpoint_state(classifier_dir)

  classifier_saver_1 = tf.train.Saver(var_list = classifier_vars_1)
  classifier_saved_model_1 = tf.train.get_checkpoint_state(classifier_dir_1)

  with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    ae_saver.restore(sess,ae_saved_model.model_checkpoint_path)
    classifier_saver.restore(sess,classifier_saved_model.model_checkpoint_path)
    classifier_saver_1.restore(sess,classifier_saved_model_1.model_checkpoint_path)

    with tf.contrib.slim.queues.QueueRunners(sess):
        for _ in range(10):
#            data = sess.run(predictions_['features.source_ids'])
            _, d_loss, r, f = sess.run([train_op,total_loss,real_acc,fake_acc])
            print ("Loss: {:.2f}, real: {:.2f}, fake: {:.2f}".format(d_loss,r,f))
#            print (translate(data[0]))
#            data = preds['features.source_ids']
#            print (preds['features.source_tokens'][0])
#            print (infer_classifier(cnn,sess,data,classifier_dir,128))
#      print (len(out))
#      print (type(loss_),type(predictions_))

def main(_argv):
  """The entrypoint for the script"""

  # Parse YAML FLAGS
  FLAGS.hooks = _maybe_load_yaml(FLAGS.hooks)
  FLAGS.metrics = _maybe_load_yaml(FLAGS.metrics)
  FLAGS.model_params = _maybe_load_yaml(FLAGS.model_params)
  FLAGS.input_pipeline_train = _maybe_load_yaml(FLAGS.input_pipeline_train)
  FLAGS.input_pipeline_dev = _maybe_load_yaml(FLAGS.input_pipeline_dev)

  # Load flags from config file
  final_config = {}
  if FLAGS.config_paths:
    for config_path in FLAGS.config_paths.split(","):
      config_path = config_path.strip()
      if not config_path:
        continue
      config_path = os.path.abspath(config_path)
      tf.logging.info("Loading config from %s", config_path)
      with gfile.GFile(config_path.strip()) as config_file:
        config_flags = yaml.load(config_file)
        final_config = _deep_merge_dict(final_config, config_flags)

  tf.logging.info("Final Config:\n%s", yaml.dump(final_config))

  # Merge flags with config values
  for flag_key, flag_value in final_config.items():
    if hasattr(FLAGS, flag_key) and isinstance(getattr(FLAGS, flag_key), dict):
      merged_value = _deep_merge_dict(flag_value, getattr(FLAGS, flag_key))
      setattr(FLAGS, flag_key, merged_value)
    elif hasattr(FLAGS, flag_key):
      setattr(FLAGS, flag_key, flag_value)
    else:
      tf.logging.warning("Ignoring config flag: %s", flag_key)

  if FLAGS.save_checkpoints_secs is None \
    and FLAGS.save_checkpoints_steps is None:
    FLAGS.save_checkpoints_secs = 600
    tf.logging.info("Setting save_checkpoints_secs to %d",
                    FLAGS.save_checkpoints_secs)

  if not FLAGS.output_dir:
    FLAGS.output_dir = tempfile.mkdtemp()

  if not FLAGS.input_pipeline_train:
    raise ValueError("You must specify input_pipeline_train")

  if not FLAGS.input_pipeline_dev:
    raise ValueError("You must specify input_pipeline_dev")

  train_options = training_utils.TrainOptions(
      model_class=FLAGS.model,
      model_params=FLAGS.model_params)

  create_experiment(FLAGS.output_dir)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
