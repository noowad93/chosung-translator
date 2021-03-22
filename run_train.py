import torch
import logging
import sys
from typing import Tuple
import math
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from torch.optim.adam import Adam
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from chosung_translator.config import TrainConfig
from chosung_translator.data import ChosungTranslatorDataset
from chosung_translator.utils import load_data


def train(
    config: TrainConfig,
    model: BartForConditionalGeneration,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    optimizer: Adam,
    logger: logging.Logger,
    device=torch.device,
):
    """ 지정된 Epoch만큼 모델을 학습시키는 함수입니다. """
    model.to(device)
    global_step = 0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        loss_sum = 0.0
        for data in train_dataloader:
            global_step += 1
            data = _change_device(data, device)
            optimizer.zero_grad()
            output = model.forward(
                input_ids=data[0],
                attention_mask=data[1],
                decoder_input_ids=data[2],
                labels=data[3],
                decoder_attention_mask=data[4],
                return_dict=True,
            )
            loss = output["loss"]
            loss.backward()
            loss_sum += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step % config.train_log_interval == 0:
                mean_loss = loss_sum / config.train_log_interval
                logger.info(
                    f"Epoch {epoch} Step {global_step} " f"Loss {mean_loss:.4f} Perplexity {math.exp(mean_loss):8.2f}"
                )
                loss_sum = 0.0
            if global_step % config.dev_log_interval == 0:
                _validate(model, dev_dataloader, logger, device)
            if global_step % config.save_interval == 0:
                model.save_pretrained(f"{config.save_model_file_prefix}_{global_step}")


def _validate(
    model: BartForConditionalGeneration,
    dev_dataloader: DataLoader,
    logger: logging.Logger,
    device: torch.device,
):
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for data in tqdm(dev_dataloader):
            data = _change_device(data, device)
            output = model.forward(
                input_ids=data[0],
                attention_mask=data[1],
                decoder_input_ids=data[2],
                labels=data[3],
                decoder_attention_mask=data[4],
                return_dict=True,
            )
            loss = output["loss"]
            loss_sum += loss.item()
    mean_loss = loss_sum / len(dev_dataloader)
    logger.info(f"[Validation] Loss {mean_loss:.4f} Perplexity {math.exp(mean_loss):8.2f}")
    model.train()


def _change_device(data: Tuple[torch.Tensor, ...], device: torch.device):
    return tuple((data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)))


def _save_model(model: nn.Module, save_model_file_prefix: str, step: int):
    """ 모델을 지정된 경로에 저장하는 함수입니다. """
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), f"{save_model_file_prefix}_step_{step}.pth")
    else:
        torch.save(model.state_dict(), f"{save_model_file_prefix}_step_{step}.pth")


def main():
    # Config
    config = TrainConfig()

    # Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data Loading...
    raw_train_instances = load_data(config.train_file_path)
    raw_dev_instances = load_data(config.dev_file_path)
    logger.info(f"훈련용 예시 개수:{len(raw_train_instances)}\t 검증용 예시 개수:{len(raw_dev_instances)}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.pretrained_model_name)

    train_dataset = ChosungTranslatorDataset(raw_train_instances, tokenizer, config.max_seq_len)
    dev_dataset = ChosungTranslatorDataset(raw_dev_instances, tokenizer, config.max_seq_len)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = BartForConditionalGeneration.from_pretrained(config.pretrained_model_name)

    # Train
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    train(config, model, train_dataloader, dev_dataloader, optimizer, logger, device)


if __name__ == "__main__":
    main()
