# -*- coding: utf-8 -*-
import string

ALL_UNICODE_CHARS = string.whitespace + string.punctuation + string.digits + \
    u'bcdđfghjklmnpqrstvwxz' + u'aáàảãạâấầẩẫậăắằẳẵặeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọơớờởỡợôốồổỗộuúùủũụưứừửữựyýỳỷỹỵ'

# Sentiment values
POSITIVE = 1
NEGATIVE = -1
NEUTRAL = 0

# File path
DATA_PATH = 'data/'
MODEL_PATH = 'models/'
MONITOR_PATH = 'tensorboard/'

POSITIVE_DATA_PATH = DATA_PATH + 'positive_dataset.csv'
NOT_POSITIVE_DATA_PATH = DATA_PATH + 'not_positive_dataset.csv'
NEGATIVE_DATA_PATH = DATA_PATH + 'negative_dataset.csv'
NOT_NEGATIVE_DATA_PATH = DATA_PATH + 'not_negative_dataset.csv'
TEST_DATA_PATH = DATA_PATH + 'test_dataset.csv'
NEGATIVE_TRAINING_HDF5_PATH = DATA_PATH + 'negative_training.hdf5'
POSITIVE_TRAINING_HDF5_PATH = DATA_PATH + 'positive_training.hdf5'

NEGATIVE_CHECKPOINTS_PATH = MODEL_PATH + 'negative/checkpoints'
POSITIVE_CHECKPOINTS_PATH = MODEL_PATH + 'positive/checkpoints'

NEGATIVE_TRAIN_SUMMARY_PATH = MONITOR_PATH + 'train_negative/train'
NEGATIVE_DEV_SUMMARY_PATH = MONITOR_PATH + 'train_negative/dev'
POSITIVE_TRAIN_SUMMARY_PATH = MONITOR_PATH + 'train_positive/train'
POSITIVE_DEV_SUMMARY_PATH = MONITOR_PATH + 'train_positive/dev'

WORD_EMBEDDING_DATASET_PATH = DATA_PATH + 'word_embedding_dataset.csv'
VOCAB_PATH = MODEL_PATH + 'sentiment_vocab.bin'
CHAR_VOCAB_PATH = MODEL_PATH + 'char_sentiment_vocab.bin'
WORD_EMBEDDING_PATH = MODEL_PATH + 'sentiment_word_embedding.bin'

SHORT_TXT_NEGATIVE_TRAINING_HDF5_PATH = DATA_PATH + 'short_txt_negative_training.hdf5'
SHORT_TXT_NEGATIVE_CHECKPOINTS_PATH = MODEL_PATH + 'short_txt_negative/checkpoints'
SHORT_TXT_NEGATIVE_TRAIN_SUMMARY_PATH = MONITOR_PATH + 'train_short_txt_negative/train'
SHORT_TXT_NEGATIVE_DEV_SUMMARY_PATH = MONITOR_PATH + 'train_short_txt_negative/dev'
SHORT_TXT_VOCAB_PATH = MODEL_PATH + 'short_txt_sentiment_vocab.bin'


# Text properties
MAX_WORD_LEN = 100
SHORT_TXT_MAX_WORD_LEN = 10
SHORT_TXT_MAX_CHAR_LEN = 25

# Common DL params
MAX_POOL_SIZE = 5
WORD_EMBEDDING_SIZE = 150
SHORT_TXT_MAX_BATCH_SIZE = 256