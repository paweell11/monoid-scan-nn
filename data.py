import numpy as np
import unicodedata
import urllib.request

DATA_URL = "https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt"

SEQ_LEN = 128                      # T - długość jednej sekwencji
TRAIN_SPLIT = 0.90                 
VAL_SPLIT = 0.10                 

RNG_SEED = 42                      


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00A0", " ")    
    return s


def build_vocab(train_text: str):
    chars = sorted(list(set(train_text)))
    char2id = {ch: i for i, ch in enumerate(chars)}
    id2char = chars
    return char2id, id2char


def encode_text(s: str, char2id: dict) -> np.ndarray:
    arr = np.fromiter((char2id[ch] for ch in s), dtype=np.int32)
    return arr  # (N,)


def make_window_starts(num_tokens: int, T: int) -> np.ndarray:
    last_start = num_tokens - (T + 1)
    if last_start < 0:
        return np.array([], dtype=np.int64)
    return np.arange(0, last_start + 1, T, dtype=np.int64)


def batch_iterator(ids: np.ndarray, seq_len: int, batch_size: int, shuffle: bool, seed: int):
    rng = np.random.default_rng(seed)

    while True:
        starts = make_window_starts(len(ids), seq_len)

        if shuffle:
            rng.shuffle(starts)

        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i:i + batch_size]

            # jeśli ostatni batch jest niepełny pomijam go
            if len(batch_starts) < batch_size:
                continue

            # zbuduj x i y
            x_list = []
            y_list = []
            for st in batch_starts:
                chunk = ids[st:st + seq_len + 1]     # długość T+1
                x_seq = chunk[:-1]                   # pierwsze T
                y_seq = chunk[1:]                    # przesunięte o 1
                x_list.append(x_seq)
                y_list.append(y_seq)

            x = np.stack(x_list, axis=0).astype(np.int32)  # (B,T)
            y = np.stack(y_list, axis=0).astype(np.int32)  # (B,T)

            yield x, y


class CharData:
    def __init__(self):
        self.char2id = None
        self.id2char = None
        self.train_ids = None
        self.val_ids = None

    def _load_raw_text(self):
        with urllib.request.urlopen(DATA_URL) as resp:
            raw_bytes = resp.read()
        raw_text = raw_bytes[100:-1668].decode("utf-8", errors="ignore") 
        text = normalize_text(raw_text)
        return text

    def _split_train_val(self, full_text: str):
        n = len(full_text)
        n_train = int(n * TRAIN_SPLIT)
        train_text = full_text[:n_train]
        val_text = full_text[n_train:]
        return train_text, val_text

    def prepare(self):
        # pobranie całego tekstu
        all_text = self._load_raw_text()

        # podział na train / val 
        train_text, val_text = self._split_train_val(all_text)

        # budowa słownika 
        self.char2id, self.id2char = build_vocab(all_text)

        # zmiana tekstu na sekwencje ID
        self.train_ids = encode_text(train_text, self.char2id)  
        self.val_ids = encode_text(val_text,   self.char2id)  

    def train_loader(self, batch_size: int, shuffle: bool = True):
        assert self.train_ids is not None, "Najpierw wywołaj prepare()."
        return batch_iterator(
            self.train_ids, seq_len=SEQ_LEN, batch_size=batch_size, shuffle=shuffle, seed=RNG_SEED,
        )

    def val_loader(self, batch_size: int, shuffle: bool = False):
        assert self.val_ids is not None, "Najpierw wywołaj prepare()."
        return batch_iterator(
            self.val_ids, seq_len=SEQ_LEN, batch_size=batch_size, shuffle=shuffle, seed=RNG_SEED,
        )

    def vocab_size(self) -> int:
        return len(self.id2char)

    def encode_str(self, s: str) -> np.ndarray:
        return encode_text(s, self.char2id)

    def decode_ids(self, ids_array) -> str:
        return "".join(self.id2char[int(i)] for i in ids_array)
