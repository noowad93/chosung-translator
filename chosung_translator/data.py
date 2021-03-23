from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from chosung_translator.utils import convert_text_to_chosung


class ChosungTranslatorDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizerFast, max_seq_len: int = 48):
        self.texts = texts
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        chosung_text = convert_text_to_chosung(self.texts[index])

        tokenized_chosung_text = self.tokenizer.tokenize(chosung_text)
        encoder_input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_chosung_text)
        encoder_input_ids = encoder_input_ids[: self.max_seq_len - 1]

        tokenized_text = self.tokenizer.tokenize(self.texts[index])
        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        token_ids = token_ids[: self.max_seq_len - 1]

        decoder_input_ids = [self.tokenizer.bos_token_id] + token_ids
        decoder_output_ids = token_ids + [self.tokenizer.eos_token_id]

        padded_encoder_input_ids = torch.tensor(
            encoder_input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(encoder_input_ids)),
            dtype=torch.long,
        )
        padded_decoder_input_ids = torch.tensor(
            decoder_input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(decoder_input_ids)),
            dtype=torch.long,
        )
        padded_decoder_output_ids = torch.tensor(
            decoder_output_ids + [-100] * (self.max_seq_len - len(decoder_output_ids)),
            dtype=torch.long,
        )
        encoder_attention_mask = torch.tensor(
            [1] * len(encoder_input_ids) + [0] * (self.max_seq_len - len(encoder_input_ids)), dtype=torch.long
        )
        decoder_attention_mask = torch.tensor(
            [1] * len(decoder_input_ids) + [0] * (self.max_seq_len - len(decoder_input_ids)), dtype=torch.long
        )
        return tuple(
            (
                padded_encoder_input_ids,
                encoder_attention_mask,
                padded_decoder_input_ids,
                padded_decoder_output_ids,
                decoder_attention_mask,
            )
        )

    def __len__(self):
        return len(self.texts)
