import numpy as np
import tensorflow as tf

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,
            name_scope='TextCNN'):

            self.embedding_size = embedding_size
            self.sequence_length = sequence_length
            self.num_classes = num_classes 
            self.vocab_size = vocab_size 
            self.filter_sizes = filter_sizes 
            self.num_filters = num_filters

            # Keeping track of l2 regularization loss (optional)
            self.l2_loss = tf.constant(0.0)
            self.l2_reg_lambda = l2_reg_lambda
            self.name_scope=name_scope

    def inference(self,input_x, dropout_keep_prob = tf.constant(1.0)):
        with tf.variable_scope(self.name_scope):
            # Embedding layer
                with tf.device('/cpu:0'), tf.name_scope("embedding"):
                    W = tf.Variable(
                            tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                            name="W")
                    embedded_chars = tf.nn.embedding_lookup(W, input_x)
                    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

                # Create a convolution + maxpool layer for each filter size
                pooled_outputs = []
                for i, filter_size in enumerate(self.filter_sizes):
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        # Convolution Layer
                        filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                        b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                        conv = tf.nn.conv2d(
                                embedded_chars_expanded,
                                W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        # Maxpooling over the outputs
                        pooled = tf.nn.max_pool(
                                h,
                                ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name="pool")
                        pooled_outputs.append(pooled)

                # Combine all the pooled features
                num_filters_total = self.num_filters * len(self.filter_sizes)
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

                # Add dropout
                with tf.name_scope("dropout"):
                    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

                # Final (unnormalized) scores and predictions
                with tf.name_scope("output"):
                    W = tf.get_variable(
                            "W",
                            shape=[num_filters_total, self.num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
                    self.l2_loss += tf.nn.l2_loss(W)
                    self.l2_loss += tf.nn.l2_loss(b)
                    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
                    predictions = tf.argmax(scores, 1, name="predictions")

                    return scores, predictions

    def loss(self,scores,input_y):
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
            loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            return loss

    def accuracy(self,predictions,input_y):
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            return accuracy
