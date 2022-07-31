#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from absl import app, flags
from os import cpu_count
from pytorch_lightning import LightningModule, Trainer
from transformers import BertForSequenceClassification, BertTokenizerFast, DataCollatorWithPadding
from datasets import load_dataset


flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_string('dataset', 'ag_news', '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('gpus', 1, '')
flags.DEFINE_float('lr', 0.0001, '')
FLAGS = flags.FLAGS


class SentimentClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(FLAGS.model, num_labels=4)
        # Freeze Bert, train only the classification head
        # self.model.bert.requires_grad_(False)

    def prepare_data(self):
        # Prepare datasets
        tokenizer = BertTokenizerFast.from_pretrained(FLAGS.model)
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def tokenize(x):
            # Rename labels field
            x['labels'] = x['label']
            # We'll pad later dynamically in batches
            x['input_ids'] = tokenizer(
                x['text'],
                truncation=True,
                max_length=tokenizer.model_max_length)['input_ids']
            return x

        self.train_dataset = load_dataset(FLAGS.dataset, split='train[:5%]').map(
            tokenize, batched=True, remove_columns=['text', 'label'])
        self.train_dataset.set_format(type='torch')
        self.test_dataset = load_dataset(FLAGS.dataset, split='test[:5%]').map(
            tokenize, batched=True, remove_columns=['text', 'label'])
        self.test_dataset.set_format(type='torch')

    def train_dataloader(self):
        # Return training dataloader
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=FLAGS.batch_size,
            drop_last=True,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=cpu_count()
        )
        return train_loader

    def val_dataloader(self):
        # Return validation dataloader
        val_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=FLAGS.batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=cpu_count()
        )
        return val_loader

    def configure_optimizers(self):
        # Return optimizer
        opt = torch.optim.Adam(self.parameters(), lr=FLAGS.lr)
        return opt

    def forward(self, input_ids, attention_mask):
        # Used only in inference
        out = self.model(input_ids, attention_mask)
        return out.logits

    def training_step(self, batch, batch_idx):
        # Compute and return loss
        logits = self.forward(batch['input_ids'], batch['attention_mask'])
        # Note that `cross_entropy` expects logits
        loss = F.cross_entropy(logits, batch['labels']).mean()
        acc = (logits.argmax(-1) == batch['labels']).float()
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        # Called at the end of an epoch
        loss = torch.cat([out['loss'].unsqueeze(-1) for out in outputs]).mean()
        acc = torch.cat([out['acc'] for out in outputs]).mean()
        self.log('loss', {'train': loss})
        self.log('acc', {'train': acc})

    def validation_step(self, batch, batch_idx):
        # Compute loss on validation
        logits = self.forward(batch['input_ids'], batch['attention_mask'])
        # Note that `cross_entropy` expects logits
        loss = F.cross_entropy(logits, batch['labels']).mean()
        acc = (logits.argmax(-1) == batch['labels']).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        # Called at the end of an epoch
        loss = torch.cat([out['loss'].unsqueeze(-1) for out in outputs]).mean()
        acc = torch.cat([out['acc'] for out in outputs]).mean()
        self.log('loss', {'val': loss})
        self.log('acc', {'val': acc})


def main(_):
    model = SentimentClassifier()
    trainer = Trainer(
        max_epochs=FLAGS.epochs,
        gpus=FLAGS.gpus
    )
    trainer.fit(model)


if __name__ == '__main__':
    app.run(main)
