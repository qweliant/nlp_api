from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline
)

import re

sting  = " One day I said to my self in my own thought ‘whom am I praying to or is  there  a  God  who  listens  to  me?’  At  this  thought  I  was  invaded  by  dead full sadness and I said: \
‘In vain have I kept my own heart pure (as David says). Later on I thought of the words of the same David, ‘Is the  inventor  of  the  ear  unable  to  hear?’ \
and  I  said:  ‘who  is  it  thatprovided  me  with  an  ear  to  hear,  who  created  me  as  a  rational[being] and how have I come into this world? \
Where do I come from? Had  I  lived  before  the  creator  of  the  world,  I  would  have  known  the  beginning of my life and of the consciousness [of myself] \
that created me? Was I created by my own hands? But I didn’t exist before I was created. If I say that my father and my mother created me, \
then I must search for the creator of my parents and of the parents of my parents until they arrive at the first who were not created as we [are] \
but who came into this world in some other way without being generated. For if  they  themselves  have  been  created,  I  know  nothing  of  their  origin  unless I say, ‘he who created them from nothing most be an uncreated 2 \
essence  who  is  and  will  be  for  all  centuries  [to  come]  the  lord  and  master  of  all  things,  without  beginning  or  end,  immutable,  whose  years cannot be numbered.’ \
And I said: ‘Therefore, there is a creator; else there would have been no creation. This creator who endowed us with the gifts of intelligence and reason, \
cannot he himself be without them?  For  he  created  us  as  intelligent  beings  from  the  abundance  of  this intelligence and the same one being comprehends all, \
creates all,is  almighty.’  And  I  used  to  say:  ‘my  creator  will  hear  me  if  I  pray  to  him,’  and  because  of  this  thought  I  felt  very  happy"

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
            inputs, max_length=800, do_sample=True, top_p=0.95, top_k=111
        )
        self.generated = str(prompt + self.tokenizer.decode(outputs[0])[prompt_length:])
        self.generated = re.sub("[^A-Za-z0-9 ]+", "", self.generated)
        return self.generated

    def summarize(self, text=sting):

        if self.generated:
            summarizer = pipeline("summarization")
            v = summarizer(self.generated, max_length=250, min_length=30)[0]["summary_text"]
            return v
        
        summarizer = pipeline("summarization")
        v = summarizer(text, max_length=250, min_length=30)[0]["summary_text"]

        return v

    def save_model(self):
        pass
