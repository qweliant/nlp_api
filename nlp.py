from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline
)

import re

sting  = " One day I said to my self in my own thought ‘whom am I praying to or is  there  a  God  who  listens  to  me?’  At  this  thought  I  was  invaded  by  dead full sadness"

class NLP:
    def __init__(self):
        
        self.model = GPT2LMHeadModel.from_pretrained("./models/")
        self.tokenizer = GPT2Tokenizer.from_pretrained("./models/")  # Add specific options if needed
        self.generated = None

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
            inputs, max_length=225, do_sample=True, top_p=0.95, top_k=111
        )
        self.generated = str(prompt + self.tokenizer.decode(outputs[0])[prompt_length:])
        self.generated = re.sub("[^A-Za-z0-9 ]+", "", self.generated)
        return self.generated

    def summarize(self, text=sting):

        if self.generated is None:
            summarizer = pipeline("summarization")
            v = summarizer(text, max_length=80, min_length=30)[0]["summary_text"]
            return v
        
        summarizer = pipeline("summarization")
        v = summarizer(self.generated, max_length=80, min_length=30)[0]["summary_text"]
        return v

    def save_model(self):
        pass
