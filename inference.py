import pandas as pd
import spacy
from tqdm import tqdm
# Use a pipeline as a high-level helper
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
from peft import PeftModel, LoraConfig

LANG_PROMPTS = {
    'zh': '排毒',
    'es': 'Desintoxicar:',
    'ru': 'Детоксифицируй:',
    'ar': 'إزالة السموم:',
    'hi': 'विषहरण:',
    'uk': 'Детоксифікуй:',
    'de': 'Entgiften:',
    'am': 'መርዝ መርዝ:',
    'en': 'Detoxify:',
}

def detoxify(text, toxic_text, lang, model, tokenizer):
    encodings = tokenizer(LANG_PROMPTS[lang] + text, None, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(**encodings.to(model.device))
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


MODEL_PATH = 'checkpoints/checkpoint_5000'

config = LoraConfig.from_pretrained(MODEL_PATH)

model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/chatbot_models2/alignment/aya-101").to('cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.add_special_tokens({'additional_special_tokens': ['<mask>']})
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, MODEL_PATH, config=config, device='cpu').to('cpu')

text = "пошел нахуй чмо"
# masked_text = "Детоксикация путем замены <mask>: Вот бы тебя <mask>, <mask>"

encodings = tokenizer(LANG_PROMPTS["ru"] + text, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(**encodings.to(model.device), max_new_tokens=64,
                            #  do_sample=True, temperature=1., top_p=0.95,
                             num_beams=5, repetition_penalty=1.1
                             )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))