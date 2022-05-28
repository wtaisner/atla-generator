import pandas as pd
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
