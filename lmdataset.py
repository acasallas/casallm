import numpy as np, torch, os

class IndexedLMDataset(torch.utils.data.Dataset):
    def __init__(self, token_path, index_path, dtype=np.uint32):
        self.token_path = token_path
        self.index_path = index_path
        self.dtype = dtype
        self.seq = 1025

        # Lazily open memmaps in worker process to avoid big pickles
        self._tokens = None
        self._starts = None

    def _ensure_open(self):
        if self._tokens is None:
            nbytes = os.path.getsize(self.token_path)
            itemsize = np.dtype(self.dtype).itemsize
            length = nbytes // itemsize
            self._tokens = np.memmap(self.token_path, mode='r', dtype=self.dtype, shape=(length,))
        if self._starts is None:
            nbytes = os.path.getsize(self.index_path)
            self._starts = np.memmap(self.index_path, mode='r', dtype=np.uint64, shape=(nbytes//8,))

    def __len__(self):
        self._ensure_open()
        return self._starts.shape[0]

    def __getitem__(self, idx):
        self._ensure_open()
        s = int(self._starts[idx])
        e = s + self.seq
        seq = self._tokens[s:e]          # view on memmap, no copy
        # Convert to tensors; (important) clone to get page-locked batch if pin_memory
        x = torch.as_tensor(seq[:-1].astype(np.int64))  # inputs
        y = torch.as_tensor(seq[1:].astype(np.int64))   # labels
        return x, y