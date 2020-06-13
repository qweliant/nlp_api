from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline
)

import re


class NLP:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.generated = ""

    def generate(self, TRAIN_TEXT, prompt):
        inputs = self.tokenizer.encode(
            TRAIN_TEXT + prompt, add_special_tokens=False, return_tensors="pt"
        )

        prompt_length = len(
            self.tokenizer.decode(
                inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        )
        outputs = self.model.generate(
            inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60
        )
        self.generated = str(prompt + self.tokenizer.decode(outputs[0])[prompt_length:])
        re.sub("[^A-Za-z0-9]+", "", self.generated)
        return self.generated

    def summarize(self, text=None):

        if text:
            summarizer = pipeline("summarization")
            v = summarizer(text, max_length=250, min_length=30)[0]["summary_text"]
            return v
        
        summarizer = pipeline("summarization")
        v = summarizer(self.generated, max_length=250, min_length=30)[0]["summary_text"]

        return v

    def save_model():
        pass
