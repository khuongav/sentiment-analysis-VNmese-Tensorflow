# -*- coding: utf-8 -*-
__author__ = "Khuong Vo"

import os
import logging
import datetime
import h5py
import numpy as np
import tensorflow as tf
import constants

class WordCharRNN(object):
    """
    A Word Char RNN for text classification.
    Uses an trained word embedding layer, concatenated with Char RNN, 
    and classification by a fully connected layer.
    
    For Char RNN, we must gather different number of outputs of various records
    by using gather_nd, split the results by number of word of each records 
    (the number of splits is fixed due to the static graph), and then padded to
    the full word length
    """

    def __init__(self, word_sequence_length, char_sequence_length, loss_weight, 
                 max_batch_size, char_hidden_unit, word_hidden_unit, word_embedding_matrix, 
                 rand_char_embedding_info, rand_word_embedding_info=None, l2_reg_lambda=0.0):

        logging.info('word_sequence_length: %s', word_sequence_length)
        logging.info('char_sequence_length: %s', char_sequence_length)
        logging.info('loss_weight: %s', loss_weight)
        logging.info('max_batch_size: %s', max_batch_size)
        logging.info('char_hidden_unit: %s', char_hidden_unit)
        logging.info('word_hidden_unit: %s', word_hidden_unit)
        logging.info('l2_reg_lambda: %s', l2_reg_lambda)
        
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, word_sequence_length], name="input_x")
        self.input_x_char = tf.placeholder(tf.int32, [None, char_sequence_length], name="input_x_char")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.real_word_lens = tf.placeholder(tf.int32, [None], name="real_word_lens")
        self.real_char_lens = tf.placeholder(tf.int32, [None], name="real_char_lens")
        self.gather_indices = tf.placeholder(tf.int32, [None, 2], name="gather_indices")
        self.char_word_lens_splits = tf.placeholder(tf.int32, [max_batch_size], name="char_word_lens_splits")

        l2_loss = tf.constant(0.0)

        # Embedding layer
        if word_embedding_matrix is not None:
            word_embedding_size = word_embedding_matrix.shape[1]
            saved_embeddings = tf.constant(word_embedding_matrix)
            word_embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)
        else:
            word_vocab_size = rand_word_embedding_info['vocab_size']
            word_embedding_size = rand_word_embedding_info['embedding_size']
            word_embedding = tf.Variable(tf.random_uniform([word_vocab_size, word_embedding_size], -1.0, 1.0))

        word_seq_len_pad = tf.Variable([[0.0] * word_embedding_size], trainable=False, name='seq_len_pad')
        word_oov_pad = tf.Variable([[0.0] * word_embedding_size], trainable=False, name='oov_pad')
        word_embedding = tf.concat([word_seq_len_pad, word_embedding, word_oov_pad], 0, name="pretrained_word_embedding")
        embedded_words = tf.nn.embedding_lookup(word_embedding, self.input_x, name='word_embedding_lookup_op')

        char_vocab_size = rand_char_embedding_info['vocab_size']
        char_embedding_size = rand_char_embedding_info['embedding_size']
        char_embedding = tf.Variable(tf.random_uniform([char_vocab_size, char_embedding_size], -1.0, 1.0))
        char_seq_len_pad = tf.Variable([[0.0] * char_embedding_size], trainable=False, name='seq_len_pad')
        char_oov_pad = tf.Variable([[0.0] * char_embedding_size], trainable=False, name='oov_pad')
        char_embedding = tf.concat([char_seq_len_pad, char_embedding, char_oov_pad], 0, name="char_embedding")
        embedded_chars = tf.nn.embedding_lookup(char_embedding, self.input_x_char, name='char_embedding_lookup_op')

        rnn_fw_cell = tf.contrib.rnn.GRUCell(num_units=char_hidden_unit, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
        rnn_bw_cell = tf.contrib.rnn.GRUCell(num_units=char_hidden_unit, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
        rnn_fw_cell = tf.contrib.rnn.DropoutWrapper(rnn_fw_cell, output_keep_prob=self.dropout_keep_prob)
        rnn_bw_cell = tf.contrib.rnn.DropoutWrapper(rnn_bw_cell, output_keep_prob=self.dropout_keep_prob)
        rnn_output, cstate = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_fw_cell, cell_bw=rnn_bw_cell, inputs=embedded_chars, 
                dtype=tf.float32, sequence_length=self.real_char_lens)
        rnn_output = tf.concat(rnn_output, 2)

        gathered_rnn_output = tf.gather_nd(rnn_output, self.gather_indices)
        char_emb_splits = tf.split(gathered_rnn_output, self.char_word_lens_splits, 0)
        char_emb_splits_padded = []

        with tf.name_scope("char-word-padding"):
            for char_emb_split in char_emb_splits:
                padded_emb = tf.pad(char_emb_split, 
                                [[0, word_sequence_length - tf.shape(char_emb_split)[0]], [0, 0]], "CONSTANT")
                char_emb_splits_padded.append(padded_emb)

        concat_hidden_size = char_hidden_unit * 2
        embedded_words_by_chars = tf.stack(char_emb_splits_padded)
        embedded_words_by_chars = tf.slice(embedded_words_by_chars, [0, 0, 0], 
                                       [tf.shape(self.input_x)[0], word_sequence_length, concat_hidden_size])
        embedded_word_chars = tf.concat([embedded_words, embedded_words_by_chars], 2)
        embedded_word_chars.set_shape([None, word_sequence_length, word_embedding_size + concat_hidden_size])

        rnn_cell = tf.contrib.rnn.GRUCell(num_units=word_hidden_unit, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)

        rnn_output, wstate = tf.nn.dynamic_rnn(rnn_cell, embedded_word_chars, dtype=tf.float32)

        last = self.__last_relevant(rnn_output, self.real_word_lens)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            dense_W = tf.get_variable("dense_W", shape=[word_hidden_unit, 1], 
                            initializer=tf.contrib.layers.xavier_initializer())
            dense_b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            l2_loss += tf.nn.l2_loss(dense_W)
            l2_loss += tf.nn.l2_loss(dense_b)
            self.scores = tf.nn.xw_plus_b(last, dense_W, dense_b, name="scores")
            self.predictions = tf.round(tf.sigmoid(self.scores, name="predictions"))

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.weighted_cross_entropy_with_logits(
                logits=self.scores, targets=self.input_y, pos_weight=loss_weight)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions,"float"),name="accuracy")


    @staticmethod
    def __last_relevant(output, length):
        """
        This function get the last output before paddings of each record of a batch 
        """
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant


def _empty_dir(path):
    file_list = os.listdir(path)
    for file_name in file_list:
        os.remove(path + "/" + file_name)


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


def _load_trained_word_embedding():
    checkpoint_file = tf.train.latest_checkpoint(constants.NEGATIVE_CHECKPOINTS_PATH)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf, graph=graph)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            emb_var = [v for v in tf.trainable_variables() if v.name == "embedding/emb_W:0"][0]
    word_embedding_matrix = sess.run(emb_var)
    return word_embedding_matrix


def _get_tf_inputs(x_batch, char_x_batch, max_batch_size, space_idx=6):
    real_word_lens = [np.argmin(np.append(x, [0]))  for x in x_batch]
    real_char_lens = [np.argmin(np.append(char_x, [0]))  for char_x in char_x_batch]

    char_positions = [[i - 1 for i, x in enumerate(char_x) if x == space_idx] + 
               [np.argmin(char_x) - 1 if np.amin(char_x) == 0 else char_x.shape[0] - 1] 
               for char_x in char_x_batch]
    gather_indices = []
    for x_idx, positions in enumerate(char_positions):
        for position in positions:
            gather_indices.append([x_idx, position])

    char_word_lens_splits = [len(positions) for positions in char_positions]
    char_word_lens_splits.extend([0] * (max_batch_size - len(char_word_lens_splits)))
    return real_word_lens, real_char_lens, gather_indices, char_word_lens_splits


def train_model(
        preprocessed_data_filepath_or_buffer=constants.SHORT_TXT_NEGATIVE_TRAINING_HDF5_PATH,
        rand_word_embedding_info=None, restore_training=False, 
        checkpoint_every=None, num_epochs=None):
    """
    Train model with the defined CNNRNN architecture

    Args:
        preprocessed_data_filepath_or_buffer: preprocessed training data
        rand_word_embedding_info: dictionary contains vocab_size and embedding_size,
            used when pre-trained word embedding not in use
        restore_training: Flase to train from scratch, True to continue training
            from the last checkpoint

    Returns:
        None
    """

    # Data loading params
    dev_sample_size = 0

    # Model Hyperparameters
    dropout_keep_prob = 0.5
    char_hidden_unit = 50
    word_hidden_unit = 100
    l2_reg_lambda = 0.01
    loss_weight = 2
    rand_char_embedding_info = {'vocab_size': len(constants.ALL_UNICODE_CHARS), 'embedding_size': 75}

    # Training parameters
    batch_size = constants.SHORT_TXT_MAX_BATCH_SIZE
    if num_epochs is None:
        num_epochs = 15
    num_checkpoints = num_epochs
    summary_every = 1000

    # Misc Parameters
    allow_soft_placement = True
    log_device_placement = False

    # Data Preparation
    # ==================================================
    # Load data
    if isinstance(preprocessed_data_filepath_or_buffer, str):
        hf = h5py.File(preprocessed_data_filepath_or_buffer, 'r')
        x = np.array(hf.get('x'))
        char_x = np.array(hf.get('char_x'))
        y = np.array(hf.get('y'))
        hf.close()
    else:
        x, char_x, y = preprocessed_data_filepath_or_buffer

    all_zeros_rows = (x==0).all(1)
    x = x[~all_zeros_rows]
    char_x = char_x[~all_zeros_rows]
    y = y[~all_zeros_rows]
    
    # Randomly shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    char_x_shuffled = char_x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    if len(y) < dev_sample_size or dev_sample_size == 0:
        dev_sample_size = 0
        x_train = x_shuffled
        char_x_train = char_x_shuffled
        y_train = y_shuffled
    else:
        dev_sample_index = -1 * int(dev_sample_size)
        x_train, char_x_train = x_shuffled[:dev_sample_index], char_x_shuffled[:dev_sample_index]
        x_dev, char_x_dev = x_shuffled[dev_sample_index:], char_x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        logging.info('Train/Dev split: %s %s', len(y_train), len(y_dev))

    num_batches_per_epoch = int((len(y_train) - 1) / batch_size) + 1
    logging.info('Number of Steps per Epoch: %s', num_batches_per_epoch)
    if checkpoint_every is None:
        evaluate_every = num_batches_per_epoch
        checkpoint_every = num_batches_per_epoch
    else:
        evaluate_every = checkpoint_every

    # Load vocabulary
    if rand_word_embedding_info is None:
        word_embedding_matrix = _load_trained_word_embedding()
    else:
        word_embedding_matrix = None

    # Training
    # ==================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            word_char_rnn = WordCharRNN(
                word_sequence_length=constants.SHORT_TXT_MAX_WORD_LEN,
                char_sequence_length=constants.SHORT_TXT_MAX_CHAR_LEN,
                loss_weight=loss_weight,
                max_batch_size=batch_size,
                word_hidden_unit=word_hidden_unit,
                char_hidden_unit=char_hidden_unit,
                word_embedding_matrix=word_embedding_matrix,
                rand_word_embedding_info=rand_word_embedding_info,
                rand_char_embedding_info=rand_char_embedding_info,
                l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(word_char_rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", word_char_rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", word_char_rnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = constants.SHORT_TXT_NEGATIVE_TRAIN_SUMMARY_PATH
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph, max_queue=200)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = constants.SHORT_TXT_NEGATIVE_DEV_SUMMARY_PATH
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph, max_queue=200)

            # Checkpoint directory. Tensorflow assumes this directory already
            # exists so we need to create it
            checkpoint_dir = constants.SHORT_TXT_NEGATIVE_CHECKPOINTS_PATH
            checkpoint_prefix = constants.SHORT_TXT_NEGATIVE_CHECKPOINTS_PATH + '/model'
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, char_x_batch, y_batch):
                """
                A single training step
                """
                real_word_lens, real_char_lens, gather_indices, char_word_lens_splits = _get_tf_inputs(x_batch, char_x_batch, batch_size)
                feed_dict = {
                    word_char_rnn.input_x: x_batch,
                    word_char_rnn.input_x_char: char_x_batch,
                    word_char_rnn.input_y: y_batch,
                    word_char_rnn.dropout_keep_prob: dropout_keep_prob,
                    word_char_rnn.real_word_lens: real_word_lens,
                    word_char_rnn.real_char_lens: real_char_lens,
                    word_char_rnn.gather_indices: gather_indices,
                    word_char_rnn.char_word_lens_splits: char_word_lens_splits
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, word_char_rnn.loss, word_char_rnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                
                logging.info(
                    '%s: step %s, loss %g, acc %g',
                    time_str, step, loss, accuracy)
                if step % summary_every == 0:
                    summaries = sess.run(train_summary_op, feed_dict)
                    train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, char_x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                real_word_lens, real_char_lens, gather_indices, char_word_lens_splits = _get_tf_inputs(x_batch, char_x_batch, batch_size)
                feed_dict = {
                    word_char_rnn.input_x: x_batch,
                    word_char_rnn.input_x_char: char_x_batch,
                    word_char_rnn.input_y: y_batch,
                    word_char_rnn.dropout_keep_prob: dropout_keep_prob,
                    word_char_rnn.real_word_lens: real_word_lens,
                    word_char_rnn.real_char_lens: real_char_lens,
                    word_char_rnn.gather_indices: gather_indices,
                    word_char_rnn.char_word_lens_splits: char_word_lens_splits
                }
                step, loss, accuracy = sess.run(
                    [global_step, word_char_rnn.loss, word_char_rnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logging.info('%s: step %s, loss %g, acc %g', time_str, step, loss, accuracy)
                if writer:
                    summaries = sess.run(dev_summary_op, feed_dict)
                    writer.add_summary(summaries, step)

            # Restore training
            if restore_training:
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                # Empty training dirs
                _empty_dir(constants.SHORT_TXT_NEGATIVE_CHECKPOINTS_PATH)
                _empty_dir(constants.SHORT_TXT_NEGATIVE_TRAIN_SUMMARY_PATH)
                _empty_dir(constants.SHORT_TXT_NEGATIVE_DEV_SUMMARY_PATH)

            # Generate batches
            batches = batch_iter(list(zip(x_train, char_x_train, y_train)), batch_size, num_epochs)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, char_x_batch, y_batch = zip(*batch)
                train_step(x_batch, char_x_batch, y_batch)

                current_step = tf.train.global_step(sess, global_step)

                if current_step % evaluate_every == 0 and dev_sample_size != 0:
                    logging.info('\nEvaluation:')
                    dev_step(x_dev, char_x_dev, y_dev, writer=dev_summary_writer)
                    logging.info('')
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logging.info('Saved model checkpoint to %s\n', path)