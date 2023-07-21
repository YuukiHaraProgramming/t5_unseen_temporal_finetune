import torch
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration


class T5FineTuner(pl.LightningModule):

    def __init__(self, model_name='t5-small', lr=1e-3, dropout_rate=0.1, save_transformer_model_path=None):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, dropout_rate=dropout_rate)

        self.lr = lr
        # self.dropout = torch.nn.Dropout(0.1)  # Dropout rate of 0.1
        self.model_dir_path = save_transformer_model_path

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(*batch.values())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(*batch.values())
        self.log(f"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.print(
            f'epoch {self.current_epoch}, avg validation Loss: {self.trainer.callback_metrics["val_loss"]}')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_save_checkpoint(self, checkpoint):
        # save transformer model.
        if self.model_dir_path is not None:
            self.model.save_pretrained(self.model_dir_path)

        # save pl.LightningModule checkpoint.
        return super().on_save_checkpoint(checkpoint)
