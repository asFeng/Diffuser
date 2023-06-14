import dgl
import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import RobertaTokenizer
from transformers.utils import logging
from datasets import load_dataset,load_metric,load_from_disk
from models.diffuser_app import DiffuserForSequenceClassification
from models.diffuser_utils import DiffuserConfig
from graphtrainer import graphTrainer

logging.set_verbosity_debug()
logger = logging.get_logger()

def preprocess_function(examples):
    return tokenizer(examples['text'], padding = 'max_length', truncation=True, max_length = 1024)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1_score = metric_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": acc, "f1_score": f1_score }

metric_acc = load_metric('./metrics/accuracy')
metric_f1 = load_metric('./metrics/f1')

imdb = load_dataset("imdb")
# tokenizer = RobertaTokenizer.from_pretrained("./roberta-tokenizer", max_length = 1024)
tokenizer = RobertaTokenizer(tokenizer_file = "./roberta-tokenizer/tokenizer.json",
                             vocab_file     = "./roberta-tokenizer/vocab.json",
                             merges_file    = "./roberta-tokenizer/merges.txt",
                             max_length     = 1024)

tokenized_imdb = imdb.map(preprocess_function, batched=True)
tokenized_imdb = tokenized_imdb.map(remove_columns=["text"])
# tokenized_imdb = load_from_disk("/home")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

config = DiffuserConfig.from_json_file("./models/config.json")
config.num_labels = 2

model =  DiffuserForSequenceClassification(config = config).cuda()

training_args = TrainingArguments(
    output_dir = "./save_imdb",
    learning_rate = 3e-5,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    num_train_epochs = 20,
    weight_decay = 0.01,
    evaluation_strategy = "steps",
    eval_steps = 2000,
    logging_steps = 500,
    save_steps = 500,
    seed = 42,
    warmup_steps = 200,
    gradient_accumulation_steps = 8,
    prediction_loss_only=True
)

steps_per_train_epoch       = int(len(tokenized_imdb['train'])/(training_args.per_device_train_batch_size*torch.cuda.device_count()*training_args.gradient_accumulation_steps ))
training_args.eval_steps    = int(steps_per_train_epoch)
training_args.logging_steps = int(steps_per_train_epoch/5)
training_args.save_steps    = int(steps_per_train_epoch)

trainer = graphTrainer(
    model = model,
    config = config,
    args = training_args,
    train_dataset = tokenized_imdb["train"],
    eval_dataset = tokenized_imdb["test"],
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics = compute_metrics
)

trainer.train()
