import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
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

hf_dataset_parallel = datasets.load_from_disk('/app/data_parallel')
hf_dataset_translations = datasets.load_from_disk('/app/data_translations')

checkpoint = "/app/model"

tokenizer = AutoTokenizer.from_pretrained('bigscience/mt0-xl')

def preprocess_function_parallel(examples):
    toxic = [LANG_PROMPTS[lang] + tox for lang, tox in zip(examples['lang'], examples['toxic_sentence'])]

    inputs = tokenizer(toxic, truncation=True, max_length=128)
    labels = tokenizer(examples['neutral_sentence'], truncation=True, max_length=128)
    return {
        **inputs,
        'labels': labels.input_ids
    }

def preprocess_function_translations(examples):
    toxic = [LANG_PROMPTS[lang] + tox for lang, tox in zip(examples['lang'], examples['toxic_comment'])]

    inputs = tokenizer(toxic, truncation=True, max_length=128)
    labels = tokenizer(examples['neutral_comment'], truncation=True, max_length=128)
    return {
        **inputs,
        'labels': labels.input_ids
    }

print(hf_dataset_parallel)
print(hf_dataset_translations)

tokenized_dataset_parallel = hf_dataset_parallel.map(preprocess_function_parallel,  batched=True,)
tokenized_dataset_translations = hf_dataset_translations.map(preprocess_function_translations,   batched=True,)

logger.info("Parallel data sanity check")
logger.info(tokenized_dataset_parallel['train'][0]['input_ids'])
logger.info(tokenized_dataset_parallel['train'][0]['labels'])

logger.info(tokenizer.decode(tokenized_dataset_parallel['train'][0]['input_ids']))
logger.info(tokenizer.decode(tokenized_dataset_parallel['train'][0]['labels']))

logger.info("Translation data sanity check")
logger.info(tokenized_dataset_translations['train'][0]['input_ids'])
logger.info(tokenized_dataset_translations['train'][0]['labels'])

logger.info(tokenizer.decode(tokenized_dataset_translations['train'][0]['input_ids']))
logger.info(tokenizer.decode(tokenized_dataset_translations['train'][0]['labels']))

train_dataset = datasets.concatenate_datasets([tokenized_dataset_translations['train'], tokenized_dataset_parallel['train']])
test_dataset = datasets.concatenate_datasets([tokenized_dataset_translations['test'], tokenized_dataset_parallel['test']])
# train_dataset = tokenized_dataset_parallel['train']
# test_dataset = tokenized_dataset_parallel['test']

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

ADD_LORA = True
# ADD_LORA = False

if ADD_LORA:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

DEVICE = 'cuda'
model.to(DEVICE)

logger.info(f"Model device: {model.device}")

# BATCH_SIZE = 8 # mt0-xl
BATCH_SIZE = 2 # mt0-xxl
OUTPUT_DIR = 'output'

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    lr_scheduler_type='cosine',
    # warmup_steps=4,
    gradient_accumulation_steps=4, # mt0-xxl
    num_train_epochs=4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    # evaluation_strategy='steps',
    # save_strategy='steps',
    # eval_steps=2500,
    # save_steps=2500,
    bf16=True,
    report_to="wandb"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR + '/model')