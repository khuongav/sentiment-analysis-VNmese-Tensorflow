# -*- coding: utf-8 -*-
__author__ = "Khuong Vo"

import logging
import multiprocessing
from gensim.models.word2vec import Word2Vec, LineSentence
import tensorflow as tf
import constants

num_cores = multiprocessing.cpu_count() - 1


def create_word_embedding(filepath_or_buffer):
    """
    Training word embedding model based on input text file or buffer

    Args:
        filepath_or_buffer: preprocessed text file path or buffer, 
            input file format is a csv file with one column, one line = one text

    Returns:
        None
    """

    sentences = LineSentence(filepath_or_buffer)
    logging.info('Training word embedding...')
    model = Word2Vec(sentences, size=constants.WORD_EMBEDDING_SIZE, window=7, 
                     min_count=500, iter=20, sg=1, workers=num_cores)
    model.save(constants.WORD_EMBEDDING_PATH)


def load_word_embedding_model():
    """
    Load trained word embedding model

    Args:
        None

    Returns:
        word embedding model
    """

    model = Word2Vec.load(constants.WORD_EMBEDDING_PATH)
    word_vectors = model.wv
    del model
    return word_vectors


def load_trained_sentiment_word_embedding(sentiment):
    if sentiment == constants.NEGATIVE:
        we_path = constants.NEGATIVE_CHECKPOINTS_PATH
    elif sentiment == constants.POSITIVE:
        we_path = constants.POSITIVE_CHECKPOINTS_PATH

    checkpoint_file = tf.train.latest_checkpoint(we_path)
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


if __name__ == "__main__":
#    data_path = constants.TRAINING_DATA_PATH + '/word_embedding_dataset_preprocessed.csv'
#    create_word_embedding(data_path)
#    w2v = load_word_embedding_model()
    model = Word2Vec.load('sentiment_word_embedding.bin')