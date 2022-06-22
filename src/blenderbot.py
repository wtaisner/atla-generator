import pandas as pd  # type: ignore
import torch
from torch.utils.data import Dataset
from typing import Any
from transformers import BlenderbotSmallTokenizer


class CustomDataset(Dataset):
    """
    Dataset for Blenderbot
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer: BlenderbotSmallTokenizer, source_len: int, resp_len: int):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.resp_len = resp_len
        self.text = self.data.response
        self.ctext = self.data.context

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length=self.source_len, padding='max_length',
                                                  return_tensors='pt', truncation=True)
        target = self.tokenizer.batch_encode_plus([text], max_length=self.resp_len, padding='max_length',
                                                  return_tensors='pt', truncation=True)

        source_ids = source['input_ids'].squeeze().to(dtype=torch.long)
        source_mask = source['attention_mask'].squeeze().to(dtype=torch.long)
        target_ids = target['input_ids'].squeeze().to(dtype=torch.long)

        lm_labels = target_ids[1:].clone().detach()  # make fast copy
        lm_labels[target_ids[1:] == self.tokenizer.pad_token_id] = -100  # replace pad tokens

        return {
            'input_ids': source_ids,
            'attention_mask': source_mask,
            'labels': lm_labels
        }


def chat_with_me(model: Any, tokenizer: BlenderbotSmallTokenizer, src_len: int = 512, steps: int = None) -> None:
    """
    chatting with trained model
    :param model: trained model
    :param tokenizer: tokenizer for given model
    :param src_len: minimal length of the source message (if longer - truncate)
    :param steps: the length of the talk (number of phrases we wish to write) (optional)
    """
    if steps is None:
        print('to quit write "quit"')
        while True:
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            user_input = input(">>User: ")
            if user_input == 'quit':
                break
            data = tokenizer.batch_encode_plus([user_input], max_length=src_len, padding='max_length',
                                               return_tensors='pt', truncation=True)

            ids = data['input_ids']
            mask = data['attention_mask']
            # generated a response while limiting the total chat history to 100 tokens,
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=100,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )

            text = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids][0]
            # pretty print last output tokens from bot
            print("Bot: {}".format(text))
    else:
        for step in range(steps):
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            user_input = input(">>User: ")
            data = tokenizer.batch_encode_plus([user_input], max_length=src_len, padding='max_length',
                                               return_tensors='pt', truncation=True)

            ids = data['input_ids']
            mask = data['attention_mask']
            # generated a response while limiting the total chat history to 100 tokens,
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=100,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )

            text = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids][0]
            # pretty print last output tokens from bot
            print("Bot: {}".format(text))
