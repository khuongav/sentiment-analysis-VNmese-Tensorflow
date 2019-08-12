# -*- coding: utf-8 -*-
__author__ = "Khuong Vo"

import unittest
from io import StringIO
from mock import patch
from word_embedding import create_word_embedding


class WordEmbeddingTest(unittest.TestCase):

    def setUp(self):
        self.data = u' iphone đẹp quá\n ' * 501


    @patch('word_embedding.Word2Vec.save')
    def test_create_word_embedding(self, word2vec_save_mock):
        word2vec_save_mock.return_value = None
        create_word_embedding(StringIO(self.data))
        self.assertTrue(word2vec_save_mock.called)


if __name__ == '__main__':
    unittest.main()
