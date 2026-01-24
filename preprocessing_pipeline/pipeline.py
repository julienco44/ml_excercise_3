import os
import sys
import json
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from text_cleaner import TextCleaner
from lemmatizer import Lemmatizer
from vocabulary import Vocabulary
from data_loader import get_dataset
from dataset import NextWordDataset


class PreprocessingPipeline:
    def __init__(
        self,
        use_lemmatization: bool = False,
        language: str = "en",
        min_sentence_length: int = 5,
        vocab_size: int = 20000,
        min_word_freq: int = 2
    ):
        self.use_lemmatization = use_lemmatization
        self.language = language
        self.min_sentence_length = min_sentence_length
        self.vocab_size = vocab_size
        self.min_word_freq = min_word_freq

        self.cleaner = TextCleaner()
        self.lemmatizer = None
        if use_lemmatization:
            self.lemmatizer = Lemmatizer(language)

        self.vocab = Vocabulary(min_freq=min_word_freq)
        self.stats = {}

    def process_texts(self, texts: List[str]) -> List[List[str]]:
        print(f"\nProcessing {len(texts)} texts...")
        print(f"Lemmatization: {'enabled' if self.use_lemmatization else 'disabled'}")

        cleaned_texts = []
        skipped_count = 0

        print("Step 1: Cleaning texts...")
        for i, text in enumerate(texts):
            cleaned = self.cleaner.clean(text)
            if cleaned and self.cleaner.is_valid(cleaned, self.min_sentence_length):
                cleaned_texts.append(cleaned)
            else:
                skipped_count += 1

            if (i + 1) % 10000 == 0:
                print(f"  Cleaned {i + 1}/{len(texts)} texts...")

        print(f"  Valid texts: {len(cleaned_texts)}")
        print(f"  Skipped: {skipped_count}")

        if self.use_lemmatization and self.lemmatizer and self.lemmatizer.is_available():
            print("\nStep 2: Lemmatizing texts...")
            sentences = self.lemmatizer.lemmatize_batch(cleaned_texts, batch_size=config.BATCH_SIZE)
        else:
            print("\nStep 2: Tokenizing texts...")
            sentences = []
            for text in cleaned_texts:
                tokens = text.split()
                # Add EOS token to mark end of document/thought
                tokens.append("<EOS>")
                sentences.append(tokens)

        # Filter by length again after tokenization/lemmatization
        sentences = [s for s in sentences if len(s) >= self.min_sentence_length]

        print(f"  Final documents: {len(sentences)}")
        self.stats["total_texts"] = len(texts)
        self.stats["valid_sentences"] = len(sentences)
        self.stats["skipped"] = skipped_count

        return sentences

    def build_vocabulary(self, sentences: List[List[str]]):
        print("\nStep 3: Building vocabulary...")
        self.vocab.build(sentences, max_size=self.vocab_size)

    def encode_sentences(self, sentences: List[List[str]]) -> List[List[int]]:
        print("\nStep 4: Encoding sentences...")
        encoded = [self.vocab.encode(s) for s in sentences]
        return encoded
    
    def chunk_sequences(self, sequences: List[List[int]], max_len: int) -> List[List[int]]:
        """
        Splits long documents into chunks of size max_len+1 (input + target).
        Overlaps could be implemented, here we do non-overlapping for simplicity.
        """
        chunked_data = []
        # Target chunk size is max_len + 1 (for input and target)
        chunk_size = max_len + 1
        
        for seq in sequences:
            # If sequence is smaller than chunk_size, we can still use it if it has at least 2 tokens
            if len(seq) < chunk_size:
                if len(seq) >= 2:
                    chunked_data.append(seq)
                continue
            
            # Create chunks
            for i in range(0, len(seq), max_len):
                chunk = seq[i : i + chunk_size]
                # Ensure we have at least 2 tokens (input+target)
                if len(chunk) >= 2:
                    chunked_data.append(chunk)
                    
        return chunked_data

    def split_data(
        self,
        sentences: List[List[str]],
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[List, List, List]:
        print("\nStep 5: Splitting data (Document level)...")

        train_sents, test_sents = train_test_split(
            sentences,
            test_size=test_size,
            random_state=config.RANDOM_SEED
        )

        train_sents, val_sents = train_test_split(
            train_sents,
            test_size=val_size,
            random_state=config.RANDOM_SEED
        )

        print(f"  Train docs: {len(train_sents)}")
        print(f"  Val docs: {len(val_sents)}")
        print(f"  Test docs: {len(test_sents)}")

        self.stats["train_size"] = len(train_sents)
        self.stats["val_size"] = len(val_sents)
        self.stats["test_size"] = len(test_sents)

        return train_sents, val_sents, test_sents

    def create_datasets(
        self,
        train_encoded: List[List[int]],
        val_encoded: List[List[int]],
        test_encoded: List[List[int]],
        max_seq_length: int = 50
    ) -> Tuple[NextWordDataset, NextWordDataset, NextWordDataset]:
        print("\nStep 6: Chunking and Creating PyTorch datasets...")
        
        # Chunk the data
        train_chunks = self.chunk_sequences(train_encoded, max_seq_length)
        val_chunks = self.chunk_sequences(val_encoded, max_seq_length)
        test_chunks = self.chunk_sequences(test_encoded, max_seq_length)

        train_dataset = NextWordDataset(train_chunks, max_seq_length)
        val_dataset = NextWordDataset(val_chunks, max_seq_length)
        test_dataset = NextWordDataset(test_chunks, max_seq_length)

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")

        self.stats["train_samples"] = len(train_dataset)
        self.stats["val_samples"] = len(val_dataset)
        self.stats["test_samples"] = len(test_dataset)

        return train_dataset, val_dataset, test_dataset

    def run(
        self,
        texts: List[str],
        output_dir: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        max_seq_length: int = 50
    ) -> Dict:
        print("=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        sentences = self.process_texts(texts)
        train_sents, val_sents, test_sents = self.split_data(sentences, test_size, val_size)

        self.build_vocabulary(train_sents)

        train_encoded = self.encode_sentences(train_sents)
        val_encoded = self.encode_sentences(val_sents)
        test_encoded = self.encode_sentences(test_sents)

        train_dataset, val_dataset, test_dataset = self.create_datasets(
            train_encoded, val_encoded, test_encoded, max_seq_length
        )

        print("\nStep 7: Saving outputs...")
        self.vocab.save(os.path.join(output_dir, "vocab.pkl"))
        train_dataset.save(os.path.join(output_dir, "train_dataset.pkl"))
        val_dataset.save(os.path.join(output_dir, "val_dataset.pkl"))
        test_dataset.save(os.path.join(output_dir, "test_dataset.pkl"))

        self.stats["vocab_size"] = len(self.vocab)
        stats_path = os.path.join(output_dir, "preprocessing_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Stats saved to {stats_path}")

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)

        return self.stats


def run_pipeline(
    dataset_name: str,
    dataset_path: str = None,
    output_dir: str = "processed_data",
    use_lemmatization: bool = False,
    language: str = "en"
) -> Dict:
    texts = get_dataset(dataset_name, dataset_path)

    pipeline = PreprocessingPipeline(
        use_lemmatization=use_lemmatization,
        language=language,
        min_sentence_length=config.MIN_SENTENCE_LENGTH,
        vocab_size=config.VOCAB_SIZE,
        min_word_freq=config.MIN_WORD_FREQ
    )

    stats = pipeline.run(
        texts=texts,
        output_dir=output_dir,
        test_size=config.TEST_SPLIT,
        val_size=config.VAL_SPLIT,
        max_seq_length=config.MAX_SEQ_LENGTH
    )

    return stats
