import json
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractproperty

from lib.data.preprocessing import text_cleanup, split_text_into_words


class VocabularyBase(ABC):
    @abstractproperty
    def unknown_token(self):
        return "<UNK>"

    @abstractproperty
    def padding_token(self):
        return "<PAD>"

    @abstractproperty
    def unknown_token_idx(self):
        return 0

    @abstractproperty
    def padding_token_idx(self):
        return 1

    @abstractmethod
    def build_vocabulary(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def vocab_size(self) -> int:
        pass


class CharacterVocabulary(VocabularyBase):
    def __init__(self):
        self.start_idx = 2
        self.char2idx = {
            self.unknown_token: self.unknown_token_idx,
            self.padding_token: self.padding_token_idx,
        }
        self.idx2char = {
            self.unknown_token_idx: self.unknown_token,
            self.padding_token_idx: self.padding_token,
        }

    @property
    def unknown_token(self):
        return super().unknown_token

    @property
    def padding_token(self):
        return super().padding_token

    @property
    def unknown_token_idx(self):
        return super().unknown_token_idx

    @property
    def padding_token_idx(self):
        return super().padding_token_idx

    def build_vocabulary(self, df: pd.DataFrame):
        idx = self.start_idx
        for text in tqdm(df["text"], desc="Building vocabulary"):
            text = text_cleanup(text)

            for c in text:
                if c in self.char2idx:
                    continue

                # character is not in char2idx
                self.char2idx[c] = idx
                self.idx2char[idx] = c

                idx += 1

    def save_vocabulary(self, path: str):
        with open(f"{path}/char2idx.json", "w") as f:
            json.dump(self.char2idx, f, indent=4)

        with open(f"{path}/idx2char.json", "w") as f:
            json.dump(self.idx2char, f, indent=4)

    def load_vocabulary(self, path: str):
        with open(f"{path}/char2idx.json") as f:
            self.char2idx = json.load(f)

        with open(f"{path}/idx2char.json") as f:
            self.idx2char = json.load(f)

    def vocab_size(self) -> int:
        return len(self.char2idx)


class WordVocabulary(VocabularyBase):
    def __init__(self):
        self.start_idx = 2
        self.word2idx = {
            self.unknown_token: self.unknown_token_idx,
            self.padding_token: self.padding_token_idx,
        }
        self.idx2word = {
            self.unknown_token_idx: self.unknown_token,
            self.padding_token_idx: self.padding_token,
        }

    @property
    def unknown_token(self):
        return super().unknown_token

    @property
    def padding_token(self):
        return super().padding_token

    @property
    def unknown_token_idx(self):
        return super().unknown_token_idx

    @property
    def padding_token_idx(self):
        return super().padding_token_idx

    def build_vocabulary(self, df: pd.DataFrame):
        idx = self.start_idx
        for text in tqdm(df["text"], desc="Building word vocabulary"):
            words = split_text_into_words(text)

            for word in words:
                if word in self.word2idx:
                    continue

                # word is not in word2idx
                self.word2idx[word] = idx
                self.idx2word[idx] = word

                idx += 1

    def save_vocabulary(self, path: str):
        with open(f"{path}/word2idx.json", "w") as f:
            json.dump(self.word2idx, f, indent=4)

        with open(f"{path}/idx2word.json", "w") as f:
            json.dump(self.idx2word, f, indent=4)

    def load_vocabulary(self, path: str):
        with open(f"{path}/word2idx.json") as f:
            self.word2idx = json.load(f)

        with open(f"{path}/idx2word.json") as f:
            self.idx2word = json.load(f)

    def vocab_size(self) -> int:
        return len(self.word2idx)


def get_vocabulary(vocabulary_type: str = "words") -> VocabularyBase:
    if vocabulary_type == "words":
        return WordVocabulary()
    elif vocabulary_type == "chars":
        return CharacterVocabulary()
    else:
        raise ValueError(f"Unknown vocabulary type: {vocabulary_type}")
