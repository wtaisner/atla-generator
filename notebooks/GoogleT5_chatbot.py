import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.GoogleT5 import chat

# MODEL PARAMETERS
MODEL = 't5-base'
#MODEL_PATH = "../outputs/GoogleT5/model_files/epoch-90"
#MODEL_PATH = "../outputs/GoogleT5/model_files/epoch-110"
MODEL_PATH = "../outputs/GoogleT5_Aang/model_files/epoch-13"

PREFIX = 'Avatar dialogue: '

MAX_SOURCE_TEXT_LENGTH = 256
MAX_TARGET_TEXT_LENGTH = 128

if __name__ == "__main__":
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = T5Tokenizer.from_pretrained(MODEL, model_max_length=MAX_SOURCE_TEXT_LENGTH)
    should_continue = True

    print("Uncle Iroh dialogue bot (write 'quit' to quit)\n")

    while True:
        user_input = input("USER: ")
        if user_input == 'quit':
            break
        model_output = chat(user_input, model, tokenizer, PREFIX, MAX_SOURCE_TEXT_LENGTH, MAX_TARGET_TEXT_LENGTH)
        print("IROH: {}".format(model_output))