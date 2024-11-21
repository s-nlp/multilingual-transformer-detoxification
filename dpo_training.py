import datasets
from transformers import AutoTokenizer, M2M100Tokenizer, BitsAndBytesConfig
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel
import torch
from trl import DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer

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

LANG_CODES = {
    'zh': 'zho_Hans',
    'es': 'spa_Latn',
    'ru': 'rus_Cyrl',
    'ar': 'arb_Arab',
    'hi': 'hin_Deva',
    'uk': 'ukr_Cyrl',
    'de': 'deu_Latn',
    'am': 'amh_Ethi',
    'en': 'eng_Latn',
}

hf_dataset = datasets.load_from_disk('data/dpo_training_beam_groups')

checkpoint_model = "bigscience/mt0-xl"
checkpoint = "models/mt0-xl/checkpoint-180000"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def preprocess_function(examples):
    toxic = [LANG_CODES[lang] + LANG_PROMPTS[lang] + tox + "</s>" for lang, tox in zip(examples['lang'], examples['toxic_comment'])]
    # masked_toxic = examples['masked_comment']
    masked_toxic = None

    inputs = tokenizer(toxic, masked_toxic, truncation=True, max_length=128, add_special_tokens=False)
    labels = tokenizer([LANG_CODES[lang] + comm + "</s>" for lang, comm in zip(examples['lang'], examples['neutral_comment'])], truncation=True, max_length=128, add_special_tokens=False)
    return {
        **inputs,
        'labels': labels.input_ids
    }

def return_prompt_and_responses(samples):
    return {
        "prompt": [
            LANG_PROMPTS[lang] + source
            for lang, source in zip(samples['lang'], samples["toxic_sentence"])
        ],
        "chosen": [s for s in samples["neutral_sentence"]],   # rated better than k
        "rejected": [s for s in samples["detoxified_sentence"]], # rated worse than j
    }

print(hf_dataset)
hf_dataset = hf_dataset.filter(lambda x: x['neutral_sentence'] != x['detoxified_sentence'])
tokenized_dataset = hf_dataset.map(return_prompt_and_responses, batched=True, remove_columns=hf_dataset['train'].column_names)
print(tokenized_dataset)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


# Load the base model.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint_model,
    # load_in_8bit=True,
    # load_in_4bit=True,
    # quantization_config=bnb_config,
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    # device_map="auto",
)
model.config.use_cache = False

# Load the adapter.
model = PeftModel.from_pretrained(
    model,
    checkpoint,
    is_trainable=True,
    # adapter_name="train",
)
# Load the adapter a second time, with a different name, which will be our reference model.
# model.load_adapter(checkpoint, adapter_name="reference")

# Initialize the trainer, without a ref_model param.
BATCH_SIZE = 4
OUTPUT_DIR = 'models/' + checkpoint + '-beam-groups'
STEPS = 5000
training_args = ORPOConfig(
    # model_adapter_name="default",
    # ref_adapter_name="reference",
    beta=0.1,
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    warmup_steps=128,
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_steps=STEPS,
    eval_steps=STEPS,
    evaluation_strategy='steps',
    # weight_decay=0.01,
    logging_steps=100,
    bf16=True,
    max_prompt_length=512,
    # max_target_length=512,
    max_completion_length=512,
    # gradient_accumulation_steps=4,
    # gradient_checkpointing=True,
    # remove_unused_columns=False,
    report_to='wandb',
)
dpo_trainer = ORPOTrainer(
    model,
    # beta=0.1,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    # max_prompt_length=512,
    # max_target_length=512,
    # data_collator=data_collator,
)

dpo_trainer.train()
dpo_trainer.save_model(OUTPUT_DIR + '/model')