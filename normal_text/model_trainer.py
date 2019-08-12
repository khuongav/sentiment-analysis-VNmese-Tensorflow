# -*- coding: utf-8 -*-
__author__ = "Khuong Vo"

import os
import shutil
import logging
import datetime
import h5py
import numpy as np
import tensorflow as tf
from common import constants
from word_embedding.word_embedding import load_word_embedding_model
from normal_text.layer_norm_gru import LayerNormGRU
SEED = 1494

class TextCNNRNN(object):
    """
    A CNN RNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling
    and fully connected layer.
    """

    def __init__(self, sequence_length, loss_weight, embedding_matrix, filter_sizes,
            num_filters, max_pool_size, hidden_unit, rand_embedding_info=None,
            l2_reg_lambda=0.0):

        logging.info('sequence_length: %s', sequence_length)
        logging.info('loss_weight: %s', loss_weight)
        logging.info('filter_sizes: %s', filter_sizes)
        logging.info('num_filters: %s', num_filters)
        logging.info('max_pool_size: %s', max_pool_size)
        logging.info('hidden_unit: %s', hidden_unit)
        logging.info('l2_reg_lambda: %s', l2_reg_lambda)

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.dropout_keep_prob1 = tf.placeholder(tf.float32, name="dropout_keep_prob1")
        self.dropout_keep_prob2 = tf.placeholder(tf.float32, name="dropout_keep_prob2")
        self.real_len = tf.placeholder(tf.int32, [None], name="real_len")
        self.panelty_matrix = tf.placeholder(tf.float32, [2, 2], name="panelty_matrix")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            if embedding_matrix is not None:
                embedding_size = embedding_matrix.shape[1]
                vocab_size = embedding_matrix.shape[0]
                saved_embeddings = tf.constant(embedding_matrix)
                self.emb_W = tf.Variable(initial_value=saved_embeddings, name="emb_W")
            else:
                vocab_size = rand_embedding_info["vocab_size"]
                embedding_size = rand_embedding_info["embedding_size"]
                self.emb_W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="emb_W")
            tf.summary.histogram("emb_W", self.emb_W)

            seq_len_pad = tf.Variable([[0.0] * embedding_size], trainable=False, name="seq_len_pad")
            oov_pad = tf.Variable([[0.0] * embedding_size], trainable=False, name="oov_pad")
            self.emb_W = tf.concat([seq_len_pad, self.emb_W, oov_pad], 0)

            self.embedded_chars = tf.nn.embedding_lookup(
                self.emb_W, self.input_x, name="embedding_lookup_op")

            self.embedded_chars_expanded = tf.expand_dims(
                self.embedded_chars, -1)

            self.embedded_chars_expanded_dropout = tf.nn.dropout(self.embedded_chars_expanded, 
                                                                 self.dropout_keep_prob1)

        # Create a convolution + maxpool layer for each filter size
        self.conv_emb_pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name="conv_emb_pad")
        reduced = np.int32(np.ceil((sequence_length) * 1.0 / max_pool_size))
        pooled_concat = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                num_prio = (filter_size - 1) // 2
                num_post = (filter_size - 1) - num_prio
                pad_prio = tf.concat([self.conv_emb_pad] * num_prio, 1)
                pad_post = tf.concat([self.conv_emb_pad] * num_post, 1)
                emb_padded = tf.concat([pad_prio, self.embedded_chars_expanded_dropout, pad_post], 1)

                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                filter_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_W")
                filter_b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="filter_b")
                tf.summary.histogram("filter_W", filter_W)
                tf.summary.histogram("filter_b", filter_b)

                conv = tf.nn.conv2d(emb_padded, filter_W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv = tf.contrib.layers.batch_norm(conv, scale=True, is_training=self.is_training)
                # Apply nonlinearity
                nonlinear_conv = tf.nn.relu(tf.nn.bias_add(conv, filter_b), name="relu")
                tf.summary.histogram("activation_conv", nonlinear_conv)

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(nonlinear_conv, ksize=[1, max_pool_size, 1, 1], 
                                strides=[1, max_pool_size, 1, 1], padding="SAME", name="pool")

                pooled = tf.reshape(pooled, [-1, reduced, num_filters])
                pooled_concat.append(pooled)

        pooled_concat = tf.concat(pooled_concat, 2)
        pooled_concat_dropout = tf.nn.dropout(pooled_concat, self.dropout_keep_prob2)

        # Create a recurrent layer for the max-pooled sequences
        with tf.name_scope("recurrent"):
            rnn_cell = LayerNormGRU(size=hidden_unit, initializer=tf.contrib.layers.xavier_initializer())

            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob2)
            rnn_output, state = tf.nn.dynamic_rnn(rnn_cell, pooled_concat_dropout, 
                                   dtype=tf.float32, sequence_length=self.real_len)
            # tf.summary.histogram("rnn_W", rnn_cell._cell.variables[0])
            # tf.summary.histogram("rnn_b", rnn_cell._cell.variables[1])

            last = self.__last_relevant(rnn_output, self.real_len)
            tf.summary.histogram("rnn_last", last)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("dense"):
            dense_W = tf.get_variable("dense_W", shape=[hidden_unit, 1], 
                            initializer=tf.contrib.layers.xavier_initializer())
            dense_b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            tf.summary.histogram("dense_W", dense_W)
            tf.summary.histogram("dense_b", dense_b)

            l2_loss += tf.nn.l2_loss(dense_W)
            l2_loss += tf.nn.l2_loss(dense_b)
            self.scores = tf.nn.xw_plus_b(last, dense_W, dense_b, name="scores")
            dense_nonlinear = tf.sigmoid(self.scores, name="predictions")
            tf.summary.histogram("activation_dense", dense_nonlinear)
            self.predictions = tf.round(dense_nonlinear)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # Panelty matrix
            gather_indices = tf.concat([tf.cast(self.predictions, tf.int32), 
                                        tf.cast(self.input_y, tf.int32)], axis = 1)
            loss_weights = tf.gather_nd(self.panelty_matrix, gather_indices)
            loss_weights = tf.expand_dims(loss_weights, -1)

            losses = tf.nn.weighted_cross_entropy_with_logits(
                logits=self.scores, targets=self.input_y, pos_weight=loss_weight)
            weighted_losses = tf.multiply(losses, loss_weights)
            self.loss = tf.reduce_mean(weighted_losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")


    @staticmethod
    def __last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant


def _empty_training_dir(sentiment):
    def empty_dir(path):
        file_list = os.listdir(path)
        for file_name in file_list:
            os.remove(path + "/" + file_name)

    if sentiment == constants.NEGATIVE:
        try:
            empty_dir(constants.NEGATIVE_CHECKPOINTS_PATH)
            constants.NEGATIVE_TRAIN_SUMMARY_PATH.split('/')
            shutil.rmtree(constants.NEGATIVE_TRAIN_SUMMARY_PATH.rsplit('/', 1)[0])
        except OSError:
            pass
    elif sentiment == constants.POSITIVE:
        try:
            empty_dir(constants.POSITIVE_CHECKPOINTS_PATH)
            shutil.rmtree(constants.POSITIVE_TRAIN_SUMMARY_PATH.rsplit('/', 1)[0])
        except OSError:
            pass



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def real_len(batches, max_pool_size):
    lens = [np.ceil(np.argmin(np.append(batch, [0])) * 1. / 
                    max_pool_size) for batch in batches]
    return lens

def __chunks(l, n):
    res = []
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
         res.append(l[i:i + n])
    return res

def train_model(sentiment, preprocessed_data_filepath_or_buffer=None, 
                rand_embedding_info=None, restore_training=False, 
                checkpoint_every=None, num_epochs=None):
    """
    Train model with the defined CNN architecture

    Args:
        rand_embedding_info: dictionary contains vocab_size and embedding_size,
            used when pre-trained word embedding not in use
        restore_training: False to train from scratch, True to continue training
            from the last checkpoint

    Returns:
        None
    """

    # Empty training dirs
    if not restore_training:
        _empty_training_dir(sentiment)

    if sentiment == constants.NEGATIVE:
        if preprocessed_data_filepath_or_buffer is None:
            preprocessed_data_filepath_or_buffer=constants.NEGATIVE_TRAINING_HDF5_PATH
        train_summary_dir = constants.NEGATIVE_TRAIN_SUMMARY_PATH
        dev_summary_dir = constants.NEGATIVE_DEV_SUMMARY_PATH
        checkpoint_dir = constants.NEGATIVE_CHECKPOINTS_PATH
    elif sentiment == constants.POSITIVE:
        if preprocessed_data_filepath_or_buffer is None:
            preprocessed_data_filepath_or_buffer=constants.POSITIVE_TRAINING_HDF5_PATH
        train_summary_dir = constants.POSITIVE_TRAIN_SUMMARY_PATH
        dev_summary_dir = constants.POSITIVE_DEV_SUMMARY_PATH
        checkpoint_dir = constants.POSITIVE_CHECKPOINTS_PATH

    # Model Hyperparameters
    filter_sizes = '3,4,5'
    num_filters = 32
    max_pool_size = constants.MAX_POOL_SIZE
    dropout_keep_prob1 = 0.75
    dropout_keep_prob2 = 0.7
    hidden_unit = 64
    l2_reg_lambda = 0
    if sentiment == constants.NEGATIVE:
        loss_weight = 5
    elif sentiment == constants.POSITIVE:
        loss_weight = 2
    learning_rate = .001

    # Training parameters
    num_checkpoints = 200
    dev_batch_size = 1500
    dev_sample_size_is = 4500
    dev_sample_size_not = 4500
    if num_epochs is None:
        num_epochs = 20
    if sentiment == constants.NEGATIVE:
        batch_size = 512
        summary_every = 200
        panelty_matrix = [[1., 1.5], [3., 1.]]
    elif sentiment == constants.POSITIVE:
        batch_size = 256
        summary_every = 150
        panelty_matrix = [[1., 1.5], [2., 1.]]

    # Misc Parameters
    allow_soft_placement = True
    log_device_placement = False

    # Data Preparation
    # ==================================================
    # Load data
    if isinstance(preprocessed_data_filepath_or_buffer, str):
        hf = h5py.File(preprocessed_data_filepath_or_buffer, 'r')
        x = np.array(hf.get('x'))
        y = np.array(hf.get('y'))
        hf.close()
    else:
        x, y = preprocessed_data_filepath_or_buffer

    all_zeros_rows = (x==0).all(1)
    x = x[~all_zeros_rows]
    y = y[~all_zeros_rows]
    
    # Randomly shuffle data
    np.random.seed(SEED)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]


    # Split train/test set
    if len(y) < dev_sample_size_is or dev_sample_size_is == 0:
        dev_sample_size = 0
        x_train = x_shuffled
        y_train = y_shuffled
    else:
        is_index = (y_shuffled==1).all(1)
        x_is = x_shuffled[is_index]
        y_is = y_shuffled[is_index]
        x_train_is, x_dev_is = x_is[dev_sample_size_is:], x_is[:dev_sample_size_is]
        y_train_is, y_dev_is = y_is[dev_sample_size_is:], y_is[:dev_sample_size_is]
    
        not_index = (y_shuffled==0).all(1)
        x_not = x_shuffled[not_index.flatten()]
        y_not = y_shuffled[not_index]
        x_train_not, x_dev_not = x_not[dev_sample_size_not:], x_not[:dev_sample_size_not]
        y_train_not, y_dev_not = y_not[dev_sample_size_not:], y_not[:dev_sample_size_not]
        
        x_train = np.concatenate([x_train_is, x_train_not])
        y_train = np.concatenate([y_train_is, y_train_not])

        x_dev = np.concatenate([x_dev_is, x_dev_not])
        y_dev = np.concatenate([y_dev_is, y_dev_not])
        
        # Remove intersections of train and dev
        nrows, ncols = x_train.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)], 'formats':ncols * [x_train.dtype]}
        intersect_indices = np.in1d(x_dev.view(dtype), x_train.view(dtype))
        x_dev = x_dev[~intersect_indices]
        y_dev = y_dev[~intersect_indices]
        logging.debug(
            'Train/Dev intersection removed: %s ', np.sum(intersect_indices))
            
        dev_sample_size = len(y_dev)
        logging.info(
            'Train/Dev split: %s %s', len(y_train), len(y_dev))

    num_batches_per_epoch = int((len(y_train) - 1) / batch_size) + 1
    logging.info('Number of Steps per Epoch: %s', num_batches_per_epoch)
    if checkpoint_every is None:
        evaluate_every = num_batches_per_epoch / 4
        checkpoint_every = num_batches_per_epoch / 4
    else:
        evaluate_every = checkpoint_every

    # Load vocabulary
    if rand_embedding_info is None:
        word_vectors = load_word_embedding_model()
        embedding_matrix = word_vectors.syn0
        embedding_size = embedding_matrix.shape[1]
    else:
        embedding_matrix = None
        embedding_size = rand_embedding_info['embedding_size']

    # Training
    # ==================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn_rnn = TextCNNRNN(
                sequence_length=constants.MAX_WORD_LEN,
                loss_weight=loss_weight,
                embedding_matrix=embedding_matrix,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=num_filters,
                max_pool_size=max_pool_size,
                hidden_unit=hidden_unit,
                rand_embedding_info=rand_embedding_info,
                l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
                train_op = optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(
                        "{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar(
                        "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            tf.summary.merge(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn_rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn_rnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge_all()

            train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_writers = []

            for i in range(int(np.ceil(dev_sample_size/float(dev_batch_size)))):
                dev_summary_writers.append(tf.summary.FileWriter(dev_summary_dir+'_'+str(i), sess.graph))

            # Checkpoint directory. Tensorflow assumes this directory already
            # exists so we need to create it
            checkpoint_prefix = checkpoint_dir + '/model'
            saver = tf.train.Saver(
                tf.global_variables(),
                max_to_keep=num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.input_y: y_batch,
                    cnn_rnn.is_training: True,
                    cnn_rnn.dropout_keep_prob1: dropout_keep_prob1,
                    cnn_rnn.dropout_keep_prob2: dropout_keep_prob2,
                    cnn_rnn.real_len: real_len(x_batch, max_pool_size),
                    cnn_rnn.conv_emb_pad: np.zeros([len(x_batch), 1, embedding_size, 1]),
                    cnn_rnn.panelty_matrix: panelty_matrix
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                
                logging.info(
                    '%s: step %s, loss %g, acc %g',
                    time_str, step, loss, accuracy)
                if step % summary_every == 0:
                    summaries = sess.run(train_summary_op, feed_dict)
                    train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.input_y: y_batch,
                    cnn_rnn.is_training: False,
                    cnn_rnn.dropout_keep_prob1: 1.0,
                    cnn_rnn.dropout_keep_prob2: 1.0,
                    cnn_rnn.real_len: real_len(x_batch, max_pool_size),
                    cnn_rnn.conv_emb_pad: np.zeros([len(x_batch), 1, embedding_size, 1]),
                    cnn_rnn.panelty_matrix: panelty_matrix
                }

                loss = sess.run([cnn_rnn.loss], feed_dict)

                if writer:
                    summaries = sess.run(dev_summary_op, feed_dict)
                    writer.add_summary(summaries, sess.run(global_step, feed_dict))

                return loss


            # Restore training
            if restore_training:
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                saver = tf.train.import_meta_graph(
                    "{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

            # Generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)), batch_size, num_epochs)

            if dev_sample_size != 0:
                test_batches = __chunks(zip(x_dev, y_dev), dev_batch_size)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)

                current_step = tf.train.global_step(sess, global_step)

                if current_step % evaluate_every == 0 and dev_sample_size != 0:
                    logging.info('\nEvaluation:')
                    for idx, test_batch in enumerate(test_batches):
                        x_test_batch, y_test_batch = zip(*test_batch)
                        dev_step(x_test_batch, y_test_batch, writer = dev_summary_writers[idx])
                    logging.info('')

                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logging.info('Saved model checkpoint to %s\n', path)
