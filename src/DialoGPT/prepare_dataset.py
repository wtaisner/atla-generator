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
    return pd.DataFrame.from_records(context, columns=columns)


class ConversationDataset(Dataset):
    """
    Dataset for DialoGPT
    """

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df = []

        for _, row in df.iterrows():
            conv = list(reversed([x for x in row]))
            self.df.append(conv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        x = self.tokenizer.batch_encode_plus(self.df[item], max_length=self.max_len, padding='max_length',
                                             return_tensors='pt', truncation=True)
        x_ids = x['input_ids'].squeeze().to(dtype=torch.long)
        x_attention = x['attention_mask'].squeeze().to(dtype=torch.long)
        return {
            'input_ids': x_ids,
            'labels': x_ids,
            'attention_mask': x_attention,
        }
