import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

RANDOM_SEED = 42
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 50
MIN_WORD_FREQ = 2
MIN_SENTENCE_LENGTH = 5

SPACY_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm"
}

BATCH_SIZE = 1000
