from typing import Any

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


class DatasetT5(Dataset):
    """
    Dataset class for GoogleT5 model.
    """
    def __init__(self, dataset: pd.DataFrame, tokenizer: PreTrainedTokenizer,  source_len: int, target_len: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len

        self.task_prefix = 'Avatar dialogue: '

        self.context = self.dataset['context']
        self.response = self.dataset['response']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        source_text = self.task_prefix + str(self.context[index])
        target_text = str(self.response[index])

        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=False
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=False
        )

        source_ids = source["input_ids"].squeeze()
        # source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        # target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            # "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            # "target_mask": target_mask.to(dtype=torch.long)
        }


def train(
        tokenizer: PreTrainedTokenizer,
        model: Any,
        device: Any,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer
):
    """
    Function to be called for training with the parameters passed from main function
    :param model: trained model, in general it should be an object of type GPT2LMHeadModel
    :param tokenizer: tokenizer for given model
    :param device: device used to store tensors
    :param loader: dataloader for validation data
    :param optimizer: optimizer to be used during training
    """
    model.train()
    loss_history = []
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        # lm_labels = y[:, 1:].clone().detach()
        # lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        # mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            # attention_mask=mask,
            labels=y
        )

        loss = outputs[0]
        loss_history.append(loss.cpu().detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Training loss: {np.mean(loss_history)}")


def validate(tokenizer: PreTrainedTokenizer, model: Any, device: torch.device, loader: DataLoader):
    """
    Function to evaluate model for predictions
    :param model: trained model, in general it should be an object of type GPT2LMHeadModel
    :param tokenizer: tokenizer for given model
    :param device: device used to store tensors
    :param loader: dataloader for validation data
    """
    model.eval()
    contexts = []
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data['source_ids'].to(device, dtype=torch.long)
            y = data['target_ids'].to(device, dtype=torch.long)
            # mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                max_length=128,
                # attention_mask=mask,
                num_beams=2,
                repetition_penalty=2.5
                )

            context = [tokenizer.decode(c, skip_special_tokens=True, clean_up_tokenization_spaces=True) for c in ids]
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

            contexts.extend(context)
            predictions.extend(preds)
            actuals.extend(target)
    return contexts, predictions, actuals


def chat(user_input: str, model: Any, tokenizer: PreTrainedTokenizer, prefix: str, source_len: int = 256,
         output_len: int = 128) -> str:
    """
    Function to generate answers from the model
    :param user_input: text input by the user
    :param model: trained model, in general it should be an object of type T5ForConditionalGeneration
    :param tokenizer: tokenizer for given model
    :param prefix: prefix used to preceed the user input
    :param source_len: maximal number of tokens in the input
    :param output_len: maximal number of tokens in the output
    :return string, models response to user's input
    """
    prefixed_input = prefix + str(user_input)
    prefixed_input = " ".join(prefixed_input.split())

    tokenized_input = tokenizer.batch_encode_plus(
        [prefixed_input],
        max_length=source_len,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=False
    )

    with torch.no_grad():
        model.eval()
        tokenized_output = model.generate(
            input_ids=tokenized_input['input_ids'],
            max_length=128,
            num_beams=2,
            repetition_penalty=2.5
        )

    text_output = tokenizer.decode(tokenized_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return text_output


def chat_with_me(model: Any, tokenizer: PreTrainedTokenizer, steps: int = None) -> None:
    """
    chatting with trained model
    :param model: trained model, in general it should be an object of type GPT2LMHeadModel
    :param tokenizer: tokenizer for given model
    :param steps: the length of the talk (number of phrases we wish to write)
    """
    PREFIX = 'Avatar dialogue: '

    MAX_SOURCE_TEXT_LENGTH = 256
    MAX_TARGET_TEXT_LENGTH = 128

    if steps is None:
        print("write 'quit' to quit early")

        while True:
            user_input = input("USER: ")
            if user_input == 'quit':
                break
            model_output = chat(user_input, model, tokenizer, PREFIX, MAX_SOURCE_TEXT_LENGTH, MAX_TARGET_TEXT_LENGTH)
            print("Bot: {}".format(model_output))
    else:
        for step in range(steps):
            user_input = input("USER: ")
            model_output = chat(user_input, model, tokenizer, PREFIX, MAX_SOURCE_TEXT_LENGTH, MAX_TARGET_TEXT_LENGTH)
            print("Bot: {}".format(model_output))
