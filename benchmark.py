import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from peft import PeftModel, LoraConfig
from loguru import logger

LANG_PROMPTS = {
    'zh': '排毒：',
    'es': 'Desintoxicar: ',
    'ru': 'Детоксифицируй: ',
    'ar': 'إزالة السموم: ',
    'hi': 'विषहरण: ',
    'uk': 'Детоксифікуй: ',
    'de': 'Entgiften: ',
    'am': 'መርዝ መርዝ: ',
    'en': 'Detoxify: ',
}

# Use a pipeline as a high-level helper
from transformers import pipeline
from sentence_transformers import SentenceTransformer

sta = pipeline("text-classification", model="textdetox/xlmr-large-toxicity-classifier")
sim = SentenceTransformer('sentence-transformers/LaBSE')

def detoxify(text, lang, model, tokenizer):
    encodings = tokenizer(LANG_PROMPTS[lang] + text, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(**encodings.to(model.device), 
                                 max_length=128,
                                 num_beams=10,
                                 no_repeat_ngram_size=3,
                                 repetition_penalty=1.2,
                                 num_beam_groups=5,
                                 diversity_penalty=2.5,
                                 num_return_sequences=5,
                                 early_stopping=True,
                                 )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def select_best_output(text, detoxs, sta, sim):
    vals = []
    for detox in detoxs:
        emb = sim.encode([text, detox], convert_to_tensor=True)

        sim_val = (emb[0] * emb[1]).sum()
        sta_val = sta(detox)[0]
        sta_score = sta_val['score']
        if sta_val['label'] == 'LABEL_1':
            sta_score = 1 - sta_score

        vals.append((detox, (sim_val*sta_score).item()))
    detox, _ = max(vals, key=lambda x: x[1])
    return detox


dataset = pd.read_csv('https://github.com/pan-webis-de/pan-code/raw/master/clef24/text-detoxification/data/sample_submission_test.tsv', sep='\t')

MODEL_PATH = "/app/model"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to('cuda') 
tokenizer = AutoTokenizer.from_pretrained('bigscience/mt0-xxl')

# USE_LORA = False
USE_LORA = True # mt0-xxl, aya
if USE_LORA:
    ADAPTER_PATH = '/app/adapter'
    config = LoraConfig.from_pretrained(ADAPTER_PATH)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, config=config, device='cuda').to('cuda')


detox = []
for i, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    # if row['lang'] != 'ru':
        # continue
    detox_text = detoxify(row['toxic_sentence'], row['lang'], model, tokenizer)
    
    detox_text = select_best_output(row['toxic_sentence'], detox_text, sta, sim)
    detox.append(detox_text)
    logger.info(f"{i} {row['toxic_sentence']} -> {detox_text}")
dataset['neutral_sentence'] = detox

dataset.to_csv('/app/output/submission.tsv', sep='\t', index=False)

if 'id' not in dataset.columns:
    dataset = dataset.reset_index(names=['id'])


input = dataset[['id', 'toxic_sentence', 'lang']].rename({'toxic_sentence': 'text'}, axis=1)
prediction = dataset[['id', 'neutral_sentence', 'lang']].rename({'neutral_sentence': 'text'}, axis=1)

input.to_json('/app/output/input.jsonl', orient='records', force_ascii=False, lines=True)
prediction.to_json('/app/output/prediction.jsonl', orient='records', force_ascii=False, lines=True)