import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset


class TemporalDataModule(pl.LightningDataModule):
    def __init__(self, data_path, tokenizer,
                 batch_size=2048, max_seq_len=256, temporal_model=False):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.temporal_model = temporal_model
        self.data_collator = default_data_collator

    def setup(self, stage=None):
        # load dataset.
        dataset = load_dataset(
            'json', data_files={'train': os.path.join(self.data_path, 'train.json'),
                                'val': os.path.join(self.data_path, 'val.json'),
                                # 'test': os.path.join(self.data_path, 'test.json'),
                                }, field='data')
        self.train_dataset, self.val_dataset = dataset['train'], dataset['val']

        # tokenization.
        self.train_dataset = self.train_dataset.map(
            self.tokenize, batched=True
        )

        self.val_dataset = self.val_dataset.map(
            self.tokenize, batched=True
        )

    def temporal_src(self, src, year):
        return f'year: {year} text: {src}'

    def tokenize(self, examples):
        # src_texts = self.temporal_src(examples['src'], examples['year']) if self.temporal_model else examples['src']
        src_texts = [self.temporal_src(src, year) for src, year in zip(examples['src'], examples['year'])] if self.temporal_model else examples['src']
        tgt_texts = examples['tgt']


        # tokenize source texts.
        tokenized_src_texts = self.tokenizer(
            src_texts, truncation=True, max_length=self.max_seq_len, padding='max_length', return_tensors='pt')

        # tokenize target text.
        tokenized_tgt_texts = self.tokenizer(
            tgt_texts, truncation=True, max_length=self.max_seq_len, padding='max_length', return_tensors='pt')

        input_ids = tokenized_src_texts['input_ids']
        attention_mask = tokenized_src_texts['attention_mask']
        labels = tokenized_tgt_texts['input_ids']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': tokenized_tgt_texts['input_ids'],
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.data_collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=self.data_collator)