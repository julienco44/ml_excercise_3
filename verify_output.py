import pickle
import sys
import os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("preprocessing_pipeline"))
from preprocessing_pipeline.vocabulary import Vocabulary
from preprocessing_pipeline.dataset import NextWordDataset

def verify():
    print("Loading vocab...")
    vocab = Vocabulary()
    vocab.load("test_output/vocab.pkl")
    print(f"Vocab size: {len(vocab)}")
    
    # Check for punctuation in vocab
    punctuations = ['.', '?', '!']
    found_punct = [p for p in punctuations if p in vocab.word2idx]
    print(f"Found punctuation in vocab: {found_punct}")
    
    # Check for EOS
    if "<EOS>" in vocab.word2idx:
        print("Found <EOS> in vocab.")
    else:
        print("ERROR: <EOS> not in vocab!")

    print("\nLoading dataset...")
    with open("test_output/train_dataset.pkl", "rb") as f:
        data = pickle.load(f)
    
    print(f"Dataset size: {len(data)}")
    
    # Check first sample
    if len(data) > 0:
        sample = data[0] # List of ints
        decoded = vocab.decode(sample)
        print(f"Sample 0 decoded: {decoded}")
        
        if "<EOS>" in decoded:
            print("Verified: <EOS> present in sample.")
        else:
            print("WARNING: <EOS> NOT found in sample (might be chunked out if long, but sample is short).")

if __name__ == "__main__":
    verify()
