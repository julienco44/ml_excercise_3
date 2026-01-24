import pickle
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class NextWordDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_len: int = 50):
        # We assume sequences are already chunked to reasonable lengths by the pipeline
        self.data = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the full sequence
        seq = self.data[idx]
        
        # Convert to tensor
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        
        # Input is sequence excluding the last token
        # Target is sequence excluding the first token (next word for each position)
        input_seq = seq_tensor[:-1]
        target = seq_tensor[1:]
        
        return input_seq, target

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.data, f)
        print(f"Dataset saved to {path} ({len(self.data)} samples)")

    @classmethod
    def load(cls, path: str) -> "NextWordDataset":
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            instance.data = pickle.load(f)
        print(f"Dataset loaded from {path} ({len(instance.data)} samples)")
        return instance


def collate_fn(batch):
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs])
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    # Targets are now sequences too, so they need padding
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded, lengths


def create_data_loader(dataset: NextWordDataset, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
