# -*- coding: utf-8 -*-
__author__ = "Khuong Vo"

import unittest
import os
import shutil
import tempfile
import numpy as np
from common import constants
from normal_text.model_trainer import train_model


class ModelTrainerTest(unittest.TestCase):

    def setUp(self):
        self.negative_checkpoints_test_path = tempfile.mkdtemp()
        self.negative_train_summary_test_path = tempfile.mkdtemp()
        self.negative_dev_summary_test_path = tempfile.mkdtemp()

        self.embedding_size = 8
        self.vocab_size = 4
        max_word_length = constants.MAX_WORD_LEN

        x = np.array([[0] * max_word_length,
                      [3] * 10 + [0] * (max_word_length - 10),
                      [3] * 3 + [2] * 2 + [0] * (max_word_length - 5),
                      [0] * max_word_length,
                      [4] * max_word_length,
                      [1] * 10 + [0] * (max_word_length - 10),
                      [5] * max_word_length])
    
        y = np.array([[0], [1], [0], [0],[0],[1],[0]])

        self.data = (x, y)

    def test_train_model(self):
        constants.NEGATIVE_TRAIN_SUMMARY_PATH = \
            self.negative_train_summary_test_path
        constants.NEGATIVE_DEV_SUMMARY_PATH = \
            self.negative_dev_summary_test_path
        constants.NEGATIVE_CHECKPOINTS_PATH = \
            self.negative_checkpoints_test_path

        rand_embedding_info = {'vocab_size': self.vocab_size,
                               'embedding_size': self.embedding_size}
        train_model(
            sentiment=constants.NEGATIVE,
            preprocessed_data_filepath_or_buffer=self.data,
            rand_embedding_info=rand_embedding_info,
            checkpoint_every=1,
            num_epochs=5)

        self.assertTrue(os.listdir(
            self.negative_checkpoints_test_path) != [])

    def tearDown(self):
        shutil.rmtree(self.negative_checkpoints_test_path)

        constants.NEGATIVE_TRAIN_SUMMARY_PATH = 'tensorboard/train_negative/train'
        constants.NEGATIVE_DEV_SUMMARY_PATH = 'tensorboard/train_negative/dev'


if __name__ == '__main__':
    unittest.main()
