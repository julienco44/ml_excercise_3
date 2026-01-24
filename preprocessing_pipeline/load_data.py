import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocabulary import Vocabulary
from dataset import NextWordDataset, create_data_loader


def load_processed_data(data_dir: str, batch_size: int = 64):
    print(f"Loading from: {data_dir}\n")

    vocab = Vocabulary()
    vocab.load(os.path.join(data_dir, "vocab.pkl"))

    train_dataset = NextWordDataset.load(os.path.join(data_dir, "train_dataset.pkl"))
    val_dataset = NextWordDataset.load(os.path.join(data_dir, "val_dataset.pkl"))
    test_dataset = NextWordDataset.load(os.path.join(data_dir, "test_dataset.pkl"))

    train_loader = create_data_loader(train_dataset, batch_size, shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size, shuffle=False)
    test_loader = create_data_loader(test_dataset, batch_size, shuffle=False)

    return vocab, train_loader, val_loader, test_loader


def inspect_data(data_dir: str, num_samples: int = 5):
    vocab, train_loader, val_loader, test_loader = load_processed_data(data_dir)

    print("\nVocabulary Stats:")
    stats = vocab.get_stats()
    print(f"  Size: {stats['vocab_size']}")
    print(f"  Most common: {[w for w, c in stats['most_common'][:10]]}")

    print(f"\nSample Data:")
    batch = next(iter(train_loader))
    inputs, targets, lengths = batch

    for i in range(min(num_samples, inputs.size(0))):
        input_seq = inputs[i][:lengths[i]].tolist()
        target = targets[i].item()
        input_words = vocab.decode(input_seq)
        target_word = vocab.idx2word.get(target, "<UNK>")
        print(f"  Input: {' '.join(input_words)}")
        print(f"  Target: {target_word}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="processed_data")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    inspect_data(args.data_dir, args.samples)
