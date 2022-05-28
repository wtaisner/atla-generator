import torch
from transformers import PreTrainedTokenizer, AutoModelForCausalLM


def chat_with_me(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizer, steps: int = 5) -> None:
    """
    chatting with trained model
    :param model: trained model
    :param tokenizer: tokenizer for given model
    :param steps: the length of the talk (number of phrases we wish to write)
    """

    chat_history_ids = None
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
        )

        # pretty print last output tokens from bot
        print("Bot: {}".format(
            tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
