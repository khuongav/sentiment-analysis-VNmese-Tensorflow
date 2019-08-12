# -*- coding: utf-8 -*-
__author__ = "Khuong Vo"

import unittest
import os
import shutil
import tempfile
import numpy as np
import constants
from short_text_model_trainer import train_model


class ModelTrainerTest(unittest.TestCase):

    def setUp(self):
        self.short_txt_negative_checkpoints_test_path = tempfile.mkdtemp()
        self.short_txt_negative_train_summary_test_path = tempfile.mkdtemp()
        self.short_txt_negative_dev_summary_test_path = tempfile.mkdtemp()

        self.embedding_size = 4
        self.vocab_size = 10
        self.max_word_len = 4
        self.max_char_len = 13

        input_x = np.array([[4,7,2,6],
                           [2,1,5,7],
                           [3,1,0,0]])
        input_x_char = np.array([[2,6,6,6,4,7,9,2,3,4,5,3,8], 
                                 [2,6,6,3,9,7,9,0,0,0,0,0,0],
                                 [3,2,6,1,1,7,1,1,1,0,0,0,0]])
        input_y = np.array([[1],[0],[1]])

        self.data = (input_x, input_x_char, input_y)

    def test_train_model(self):
        constants.SHORT_TXT_NEGATIVE_TRAIN_SUMMARY_PATH = \
            self.short_txt_negative_train_summary_test_path
        constants.SHORT_TXT_NEGATIVE_DEV_SUMMARY_PATH = \
            self.short_txt_negative_dev_summary_test_path
        constants.SHORT_TXT_NEGATIVE_CHECKPOINTS_PATH = \
            self.short_txt_negative_checkpoints_test_path
        constants.SHORT_TXT_MAX_WORD_LEN = self.max_word_len
        constants.SHORT_TXT_MAX_CHAR_LEN = self.max_char_len

        rand_word_embedding_info = {'vocab_size': self.vocab_size,
                               'embedding_size': self.embedding_size}
        train_model(
            self.data,
            rand_word_embedding_info=rand_word_embedding_info,
            checkpoint_every=1)

        self.assertTrue(os.listdir(
            self.short_txt_negative_checkpoints_test_path) != [])

    def tearDown(self):
        shutil.rmtree(self.short_txt_negative_checkpoints_test_path)
        shutil.rmtree(self.short_txt_negative_train_summary_test_path)
        shutil.rmtree(self.short_txt_negative_dev_summary_test_path)

        constants.SHORT_TXT_NEGATIVE_CHECKPOINTS_PATH = \
            'models/' + 'short_txt_negative/checkpoints'
        constants.SHORT_TXT_NEGATIVE_TRAIN_SUMMARY_PATH = \
            'tensorboard/' + 'train_short_txt_negative/train'
        constants.SHORT_TXT_NEGATIVE_DEV_SUMMARY_PATH = \
            'tensorboard/' + 'train_short_txt_negative/train'


if __name__ == '__main__':
    unittest.main()
