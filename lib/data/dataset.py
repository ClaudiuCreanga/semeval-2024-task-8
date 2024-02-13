# flake8: noqa: W503

import torch
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer, BatchEncoding, LongformerTokenizer
from torch.utils.data import Dataset

from lib.utils.constants import TruncationStrategy
from lib.data.vocabulary import CharacterVocabulary, WordVocabulary
from lib.data.preprocessing import split_text_into_words


class UtilityDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_len: int,
        out_size: int,
        number_of_chunks: int | None = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.out_size = out_size
        self.number_of_chunks = number_of_chunks

        input_ids_size = (batch_size, max_len) if number_of_chunks is None else (
            batch_size,
            number_of_chunks,
            max_len,
        )
        self.input_ids = torch.randint(low=0, high=len(tokenizer), size=input_ids_size)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.targets = torch.rand(batch_size, out_size, dtype=torch.float)
        self.corresponding_word = torch.randint(
            low=0, high=max_len, size=(batch_size, max_len)
        )

        if isinstance(tokenizer, LongformerTokenizer):
            self.targets = torch.randint(low=0, high=2, size=(batch_size, max_len))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "target": self.targets[index],
            "corresponding_word": self.corresponding_word[index],
        }


class MachineGeneratedTransformerTruncationDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        texts: np.ndarray,
        targets: np.ndarray | None,
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        truncation_strategy: str = "head_only",
    ):
        super().__init__()
        self.ids = ids
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.truncation_strategy = TruncationStrategy(truncation_strategy)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        text = self.texts[index]
        target = -1 if self.targets is None else self.targets[index]

        text, encoding = self._get_encoding(text)

        return {
            "id": item_id if isinstance(item_id, str) else torch.tensor(int(item_id)),
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "target": torch.tensor([target], dtype=torch.float),
        }

    def _get_encoding(self, text: str) -> (str, dict):
        if (
            self.truncation_strategy is None
            or self.truncation_strategy == TruncationStrategy.HEAD_ONLY
        ):
            return text, self.tokenizer(
                text,
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
        elif self.truncation_strategy == TruncationStrategy.TAIL_ONLY:
            encoding = self.tokenizer(text)
            input_ids = encoding["input_ids"]

            if len(input_ids) > self.max_len:
                input_ids = input_ids[-self.max_len:]

                # Remove the cls and sep tokens
                text = self.tokenizer.decode(input_ids[1:-1])

            return text, self.tokenizer(
                text,
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
        elif self.truncation_strategy == TruncationStrategy.HEAD_AND_TAIL:
            encoding = self.tokenizer(text)
            input_ids = encoding["input_ids"]

            if len(input_ids) > self.max_len:
                head_tokens_count = self.max_len // 4
                tail_tokens_count = 3 * head_tokens_count
                tail_tokens_count = tail_tokens_count + (
                    self.max_len - head_tokens_count - tail_tokens_count
                )

                head_input_ids = input_ids[:head_tokens_count]
                tail_input_ids = input_ids[-tail_tokens_count:]
                input_ids = head_input_ids + tail_input_ids

                # Remove the cls and sep tokens
                text = self.tokenizer.decode(input_ids[1:-1])

            return text, self.tokenizer(
                text,
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
        else:
            raise NotImplementedError(
                f"No such truncation strategy: {self.truncation_strategy}"
            )


class MachineGeneratedTransformerHierarchicalDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        texts: np.ndarray,
        tokens: BatchEncoding,
        targets: np.ndarray | None,
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        device: str,
    ):
        super().__init__()
        self.ids = ids
        self.texts = texts
        self.tokens = tokens
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        text = self.texts[index]
        target = -1 if self.targets is None else self.targets[index]

        return {
            "id": item_id if isinstance(item_id, str) else torch.tensor(int(item_id)),
            "text": text,
            "input_ids": self.tokens["input_ids"][index].to(self.device),
            "attention_mask": self.tokens["attention_mask"][index].to(self.device),
            "target": torch.tensor([target], dtype=torch.float),
        }

    @staticmethod
    def collate_fn(data):
        ids = []
        texts = []
        input_ids = []
        attention_masks = []
        targets = []

        for item in data:
            ids.append(item["id"])
            texts.append(item["text"])
            input_ids.append(item["input_ids"])
            attention_masks.append(item["attention_mask"])
            targets.append([item["target"]])

        return {
            "id": ids,
            "text": texts,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "target": torch.tensor(targets),
        }


class LongformerTokenClassificationDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        texts: np.ndarray,
        targets: np.ndarray | None,
        tokenizer: PreTrainedTokenizer | None,
        max_len: int,
        debug: bool = False,
    ):
        super().__init__()

        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None")

        self.ids = ids
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.debug = debug

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        text = self.texts[index]
        target = -1 if self.targets is None else self.targets[index]
        targets_available = False if target == -1 else True

        words = split_text_into_words(text)

        if self.debug:
            print(f"Text: {text}")
            print(f"Words: {words}")
            print(f"Machine text start position: {target}")
            print()

        targets = []
        corresponding_word = []
        tokens = []
        input_ids = []
        attention_mask = []

        for idx, word in enumerate(words):
            word_encoded = self.tokenizer.tokenize(word)  # No [CLS] or [SEP]
            sub_words = len(word_encoded)

            if targets_available:
                is_machine_text = 1 if idx >= target else 0
                targets.extend([is_machine_text] * sub_words)

            corresponding_word.extend([idx] * sub_words)
            tokens.extend(word_encoded)
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(word_encoded))
            attention_mask.extend([1] * sub_words)

            if self.debug:
                print(
                    f"word[{idx}]:\n"
                    f"{'':-<5}> tokens: {word_encoded} (no. of subwords: {sub_words})\n"
                    f"{'':-<5}> corresponding_word: {corresponding_word[-sub_words:]}\n"
                    f"{'':-<5}> input_ids: {input_ids[-sub_words:]}\n"
                    f"{'':-<5}> is_machine_text: {is_machine_text}"
                )

        if self.debug:
            print()

            print(f"corresponding_word: {corresponding_word}")
            print(f"tokens: {tokens}")
            print(f"input_ids: {input_ids}")
            print(f"attention_mask: {attention_mask}")

            print()

            print(f"Machine text start word: {words[corresponding_word[targets.index(1)]]}")
            print(f"True machine text start word: {words[target]}")

            print()

        if len(input_ids) < self.max_len - 2:
            if targets_available:
                targets = (
                    [-100]
                    + targets
                    + [-100] * (self.max_len - len(input_ids) - 1)
                )

            corresponding_word = (
                [-100]
                + corresponding_word
                + [-100] * (self.max_len - len(input_ids) - 1)
            )
            tokens = (
                [self.tokenizer.bos_token]
                + tokens
                + [self.tokenizer.eos_token]
                + [self.tokenizer.pad_token] * (self.max_len - len(tokens) - 2)
            )
            input_ids = (
                [self.tokenizer.bos_token_id]
                + input_ids
                + [self.tokenizer.eos_token_id]
                + [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids) - 2)
            )
            attention_mask = (
                [1]
                + attention_mask
                + [1]
                + [0] * (self.max_len - len(attention_mask) - 2)
            )
        else:
            if targets_available:
                targets = [-100] + targets[: self.max_len - 2] + [-100]

            corresponding_word = (
                [-100]
                + corresponding_word[: self.max_len - 2]
                + [-100]
            )
            tokens = (
                [self.tokenizer.bos_token]
                + tokens[: self.max_len - 2]
                + [self.tokenizer.eos_token]
            )
            input_ids = (
                [self.tokenizer.bos_token_id]
                + input_ids[: self.max_len - 2]
                + [self.tokenizer.eos_token_id]
            )
            attention_mask = (
                [1]
                + attention_mask[: self.max_len - 2]
                + [1]
            )

        encoded = {}
        encoded["id"] = item_id
        encoded["text"] = text
        encoded["true_target"] = torch.tensor(target)
        encoded["corresponding_word"] = torch.tensor(corresponding_word)
        encoded["input_ids"] = torch.tensor(input_ids)
        encoded["attention_mask"] = torch.tensor(attention_mask)
        if targets_available:
            encoded["target"] = torch.tensor(targets)

        if self.debug:
            print(f"Tokenized human position: {targets.index(1)}")
            print(f"Original human position: {target}")
            print(f"Full human text: {text}\n\n")

            human_truncated_text = [
                w for w in text.split(' ')[:target] if w != ''
            ]
            print(f"Human truncated text: {human_truncated_text}\n\n")

            encoded["partial_human_review"] = " ".join(human_truncated_text)

        return encoded


class TokenClassificationDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        texts: np.ndarray,
        targets: np.ndarray | None,
        char_vocabulary: CharacterVocabulary | None,
        word_vocabulary: WordVocabulary,
        char_max_len: int,
        word_max_len: int,
        debug: bool = False,
    ):
        super().__init__()

        self.ids = ids
        self.texts = texts
        self.targets = targets
        self.char_vocabulary = char_vocabulary
        self.word_vocabulary = word_vocabulary
        self.char_max_len = char_max_len
        self.word_max_len = word_max_len
        self.debug = debug

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        text = self.texts[index]
        target = -1 if self.targets is None else self.targets[index]
        targets_available = False if target == -1 else True

        words = split_text_into_words(text)

        if self.debug:
            print(f"Text: {text}")
            print(f"Words: {words}")
            print(f"Machine text start position: {target}")
            print()

        targets = []
        corresponding_word = []
        tokens = []
        input_ids = []
        char_input_ids = []
        attention_mask = []
        char_attention_mask = []

        for idx, word in enumerate(words):
            word_encoded = [c for c in word]
            sub_words = len(word_encoded)

            if targets_available:
                is_machine_text = 1 if idx >= target else 0
                targets.append(is_machine_text)
            else:
                targets.append(-100)

            corresponding_word.append(idx)
            tokens.append(word)
            input_ids.append(
                self.word_vocabulary.word2idx.get(
                    word, self.word_vocabulary.unknown_token_idx
                )
            )
            attention_mask.append(1)

            if self.char_vocabulary is not None:
                current_word_char_input_ids = [
                    self.char_vocabulary.char2idx.get(
                        c, self.char_vocabulary.unknown_token_idx
                    ) for c in word_encoded
                ]
                current_word_char_attention_mask = [1] * sub_words

                if len(current_word_char_input_ids) < self.char_max_len:
                    current_word_char_input_ids = (
                        # [self.char_vocab.padding_token_idx]
                        current_word_char_input_ids
                        + [self.char_vocabulary.padding_token_idx] * (self.char_max_len - len(current_word_char_input_ids))
                    )
                    current_word_char_attention_mask = (
                        # [1]
                        current_word_char_attention_mask
                        + [0] * (self.char_max_len - len(current_word_char_attention_mask))
                    )
                else:
                    current_word_char_input_ids = current_word_char_input_ids[: self.char_max_len]
                    current_word_char_attention_mask = current_word_char_attention_mask[: self.char_max_len]

                char_input_ids.append(current_word_char_input_ids)
                char_attention_mask.append(current_word_char_attention_mask)

            if self.debug:
                print(
                    f"word[{idx}]:\n"
                    f"{'':-<5}> tokens: {[word]}\n"
                    f"{'':-<5}> corresponding_word: {corresponding_word[-1]}\n"
                    f"{'':-<5}> input_ids: {input_ids[-1]}\n"
                    f"{'':-<5}> char_input_ids: {char_input_ids[-sub_words:]}\n"
                    f"{'':-<5}> is_machine_text: {is_machine_text}"
                )

        if self.debug:
            print()

            print(f"corresponding_word: {corresponding_word}")
            print(f"tokens: {tokens}")
            print(f"input_ids: {input_ids}")
            print(f"char_input_ids: {char_input_ids}")
            print(f"attention_mask: {attention_mask}")
            print(f"char_attention_mask: {char_attention_mask}")

            print()

            print(f"Machine text start word: {words[targets.index(1)]}")
            print(f"True machine text start word: {words[target]}")

            print()

        if len(input_ids) < self.word_max_len:
            # if targets_available:
            targets = (
                targets
                + [-100] * (self.word_max_len - len(input_ids))
            )

            corresponding_word = (
                corresponding_word
                + [-100] * (self.word_max_len - len(input_ids))
            )
            tokens = (
                tokens
                + [self.word_vocabulary.padding_token] * (self.word_max_len - len(tokens))
            )
            input_ids = (
                input_ids
                + [self.word_vocabulary.padding_token_idx] * (self.word_max_len - len(input_ids))
            )
            attention_mask = (
                attention_mask
                + [0] * (self.word_max_len - len(attention_mask))
            )

            if self.char_vocabulary is not None:
                char_input_ids = (
                    char_input_ids
                    + [[self.char_vocabulary.padding_token_idx] * self.char_max_len] * (self.word_max_len - len(char_input_ids))
                )
                char_attention_mask = (
                    char_attention_mask
                    + [[0] * self.char_max_len] * (self.word_max_len - len(char_attention_mask))
                )
        else:
            # if targets_available:
            targets = targets[: self.word_max_len]

            corresponding_word = corresponding_word[: self.word_max_len]
            tokens = tokens[: self.word_max_len]
            input_ids = input_ids[: self.word_max_len]
            attention_mask = attention_mask[: self.word_max_len]

            if self.char_vocabulary is not None:
                char_input_ids = char_input_ids[: self.word_max_len]
                char_attention_mask = char_attention_mask[: self.word_max_len]

        encoded = {}
        encoded["id"] = item_id
        encoded["text"] = text
        encoded["true_target"] = torch.tensor(target)
        encoded["corresponding_word"] = torch.tensor(corresponding_word)
        encoded["input_ids"] = torch.tensor(input_ids)
        encoded["attention_mask"] = torch.tensor(attention_mask)
        if self.char_vocabulary is not None:
            encoded["char_input_ids"] = torch.tensor(char_input_ids)
            encoded["char_attention_mask"] = torch.tensor(char_attention_mask)

        encoded["target"] = torch.tensor(targets)

        if self.debug:
            print(f"Tokenized human position: {targets.index(1)}")
            print(f"Original human position: {target}")
            print(f"Full human text: {text}\n\n")
            print(f"Human truncated text: {[w for w in text.split(' ')[:target] if w != '']}\n\n")

            encoded["partial_human_review"] = " ".join(
                [w for w in text.split(' ')[:target] if w != '']
            )

        return encoded


class MachineGeneratedNeuralNetworkStatisticsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_column: str,
        has_targets: bool,
    ):
        super().__init__()
        self.ids = df.id.to_numpy()
        self.texts = df.text.to_numpy()
        self.features = self._get_features(df=df)
        self.targets = None if not has_targets else df[label_column].to_numpy()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        text = self.texts[index]
        feature = self.features[index]
        target = -1 if self.targets is None else self.targets[index]

        return {
            "id": torch.tensor(int(item_id)),
            "text": text,
            "input_ids": torch.from_numpy(feature),
            "attention_mask": torch.tensor([1] * len(feature)),
            "target": torch.tensor([target], dtype=torch.float),
        }

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        df_features = df.copy(deep=True)

        columns = df_features.columns.to_list()

        df_features.drop(columns=["id", "text", "label"], inplace=True)
        if "model" in columns:
            df_features.drop(columns=["model"], inplace=True)
        if "source" in columns:
            df_features.drop(columns=["source"], inplace=True)
        if "clean_text" in columns:
            df_features.drop(columns=["clean_text"], inplace=True)

        return df_features.astype(np.float32).to_numpy()
