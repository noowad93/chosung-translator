import argparse
import random

import numpy as np
import torch

from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from chosung_translator.utils import convert_text_to_chosung

parser = argparse.ArgumentParser()
parser.add_argument("--finetuned-model-path", type=str, help="Path to fine-tuned model", required=True)
parser.add_argument(
    "--decoding-method",
    default="beam_search",
    type=str,
    help="Decoding method (beam_search or top_p)",
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")

    model = BartForConditionalGeneration.from_pretrained(args.finetuned_model_path)
    model.eval()
    model.to(device)

    examples = ["배고프다", "너무너무 사랑해요", "나는 너를 좋아해"]

    for example in examples:
        chosung_example = convert_text_to_chosung(example)
        # insert [SEPT] between input utterances
        input_ids = (
            torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(chosung_example))).unsqueeze(0).to(device)
        )

        if args.decoding_method == "top_p":
            outputs = model.generate(
                input_ids=input_ids,
                max_length=48,
                temperature=1.0,
                do_sample=True,
                top_p=0.8,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=10,
            )
        elif args.decoding_method == "beam_search":
            outputs = model.generate(
                input_ids=input_ids,
                max_length=48,
                num_beams=10,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=10,
            )
        else:
            raise ValueError("Enter the right decoding method (top_p or beam_search)")

        for output in outputs.tolist():
            answer = tokenizer.decode(output)
            print(f"초성: {chosung_example} \t 예측 문장: {answer}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
