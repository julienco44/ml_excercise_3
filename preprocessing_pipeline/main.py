import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline for Next Word Prediction")

    parser.add_argument(
        "--dataset",
        type=str,
        default="20newsgroups",
        choices=["20newsgroups", "fake_news", "leipzig", "plain", "directory"],
        help="Dataset to use (20newsgroups auto-downloads)"
    )

    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to dataset file or directory"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_data",
        help="Output directory for processed files"
    )

    parser.add_argument(
        "--lemmatize",
        action="store_true",
        help="Enable spaCy lemmatization"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "de"],
        help="Language for spaCy model"
    )

    parser.add_argument("--vocab-size", type=int, default=config.VOCAB_SIZE)
    parser.add_argument("--min-freq", type=int, default=config.MIN_WORD_FREQ)
    parser.add_argument("--min-length", type=int, default=config.MIN_SENTENCE_LENGTH)
    parser.add_argument("--max-seq-length", type=int, default=config.MAX_SEQ_LENGTH)
    parser.add_argument("--test-split", type=float, default=config.TEST_SPLIT)
    parser.add_argument("--val-split", type=float, default=config.VAL_SPLIT)

    return parser.parse_args()


def main():
    args = parse_args()

    config.VOCAB_SIZE = args.vocab_size
    config.MIN_WORD_FREQ = args.min_freq
    config.MIN_SENTENCE_LENGTH = args.min_length
    config.MAX_SEQ_LENGTH = args.max_seq_length
    config.TEST_SPLIT = args.test_split
    config.VAL_SPLIT = args.val_split

    print("=" * 60)
    print("NEXT WORD PREDICTION - DATA PREPROCESSING")
    print("=" * 60)
    print(f"\nDataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Lemmatization: {'enabled' if args.lemmatize else 'disabled'}")

    stats = run_pipeline(
        dataset_name=args.dataset,
        dataset_path=args.path,
        output_dir=args.output_dir,
        use_lemmatization=args.lemmatize,
        language=args.language
    )

    print("\nFinal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
