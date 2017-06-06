import tensorflow as tf

import os
dir_name = os.path.dirname(os.path.realpath(__file__))
print (dir_name)
dir(dir_name+'/seq2seq/decoders')

ae_saver = tf.train.import_meta_graph(dir_name + '/trained_models_full/polarity_full/model.ckpt-1214839.meta')

ae_graph = tf.get_default_graph()

print [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
