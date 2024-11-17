"""
    This file is used to fine-tune a BERT model on specific data
    data contains training sentences with one sentence per line
    make sure to adjust paths to files for correct processing

    The file was created on     Sat Jul   6th 2024
        it was last edited on   Thu Aug  29th 2024

    @author: Miriam S.
"""

import os
import torch
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorForLanguageModeling, BertForMaskedLM, Trainer, TrainingArguments

# TODO: change every path to correct one containing data or the cache directory before using the code
# TODO: it is further advised to run the code on GPU with enough space for checkpoint and model saving
# adjust the following path to use the correct path to create a cache directory
cache_dir = '/path/to/cache_dir/'
# uncomment two lines below if cache_dir is set correctly
# os.environ["TRANSFORMERS_CACHE"] = cache_dir
# os.environ["HF_DATASETS_CACHE"] = cache_dir

# uncomment following line and assign free server if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")

# parameters can be adapted if needed
NUM_EPOCHS = 30
BATCH_SIZE = 16
STEPS = 500_000  # checkpoints are saved every 500,000 steps, change if necessary

# load tokenizer
# 'model' can also be checkpoint in case of interruptions, in this case insert path to checkpoint
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)

# TODO: change path to correct one containing data
data = '/assign/path/to/correct_file.txt'
dataset = load_dataset('text', data_files={'data': data})

# split data into training and validation sets
train_val_dataset = dataset['data'].train_test_split(test_size=0.2)


# function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# tokenize the train and validation sets
tokenized_train_data = train_val_dataset['train'].map(tokenize_function, batched=True, remove_columns=["text"], cache_file_name=None)
tokenized_val_data = train_val_dataset['test'].map(tokenize_function, batched=True, remove_columns=["text"], cache_file_name=None)

# set_format reduces memory usage
tokenized_train_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
tokenized_val_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# data preparation for MLM purposes
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

# load pretrained model for fine-tuning
# 'model' can also be checkpoint in case of interruptions, in this case insert '/path/to/checkpoint'
model = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir=cache_dir).to(device)

# set up training arguments for hyperparameter tuning and trainer
# save_steps saves checkpoints every 'STEPS' steps
training_args = TrainingArguments(
    output_dir='./results',
    save_steps=STEPS,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_val_data
)

# train model
trainer.train()
# if resuming from checkpoint, uncomment following line, comment previous, and change path
# trainer.train(resume_from_checkpoint='/path/to/checkpoint')
# evaluate model
results = trainer.evaluate()

# save model
# TODO: change path to correct directory for final save
model.save_pretrained('/path/to/final_directory')
tokenizer.save_pretrained('/path/to/final_directory')
