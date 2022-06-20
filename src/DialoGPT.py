from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

def create_context(df: pd.DataFrame, name: str, n: int) -> pd.DataFrame:
    """
    prepares dataset for DialoGPT
    :param df: initial data containing (among others) 'character' and 'character_words' columns
    :param name: name of the character based on which the bot will be created
    :param n: length of the context
    :return: data containing responses and their context
    """
    cols = ['character', 'character_words']
    df = df[cols]
    ids = list(df[df.character == name].index)
    context = []
    for i in ids:
        row = []
        prev = i - 1 - n
        for j in range(i, prev, -1):
            row.append(df.character_words[j])
        context.append(row)

    columns = ['response', 'context']
    columns += [f'context/{i}' for i in range(n - 1)]
    return pd.DataFrame.from_records(context, columns=columns)  # type: ignore


def flatten(row: list):
    """
    flattens 2d list
    :param row: 2d list
    :return: flattened list
    """
    return [item for sublist in row for item in sublist]


class ConversationDataset(Dataset):
    """
    Dataset for DialoGPT
    """

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.examples = []

        for _, row in df.iterrows():
            conv = self._construct_conv(row)
            self.examples.append(conv)

    def _construct_conv(self, row: pd.Series):
        conv = list(reversed([self.tokenizer.encode(x) + [self.tokenizer.eos_token_id] for x in row]))
        conv = flatten(conv)
        return conv

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


def chat_with_me(model: Any, tokenizer: PreTrainedTokenizer, steps: int = 5) -> None:
    """
    chatting with trained model
    :param model: trained model, in general it should be an object of type GPT2LMHeadModel
    :param tokenizer: tokenizer for given model
    :param steps: the length of the talk (number of phrases we wish to write)
    """

    chat_history_ids = torch.zeros(1)
    for step in range(steps):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(
            bot_input_ids, max_length=200,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )  # type: ignore

        # pretty print last output tokens from bot
        print("Bot: {}".format(
            tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
