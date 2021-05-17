import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_epoch52')
    # tokenizer = get_kobart_tokenizer()
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()

file_input = open('./IWSLT/IWSLT2016.en', 'r', encoding='utf-8')
file_output = open('./IWSLT/IWSLT2016_trans_epoch52', 'w', encoding='utf-8', newline='')

lines = file_input.readlines()
for text in lines:
    if text:
        text = text.replace('\n', '')
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
	
        output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        file_output.write(output+'\n')
