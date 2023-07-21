import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model import T5FineTuner
from data import TemporalDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_name", default='t5-small')
    parser.add_argument("--file_name", default='best')

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument("--max_epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--temporal_model",
                        default=False, action='store_true', help='Add year information to training data.')
    parser.add_argument("--dropout_rate", default=0.1)

    args = parser.parse_args()

    # load pretrained model, tokenizer.
    model = T5FineTuner(model_name=args.model_name, lr=args.lr, dropout_rate=args.dropout_rate)

    # load data.
    data_module = TemporalDataModule(data_path=args.data_dir, tokenizer=model.tokenizer,
                                                batch_size=args.batch_size, max_seq_len=args.max_seq_len,
                                                temporal_model=args.temporal_model
                                                )

    # early stopping.
    early_stop_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.0001, patience=2, verbose=True, mode='min')

    # checkpoint saving.
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.exp_dir,
        filename=args.file_name,
        save_top_k=1,
        mode='min'
        )

    # training settings.
    trainer = pl.Trainer(
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=args.max_epochs,
        deterministic=True,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        # devices=[0, 1, 2, 3],
        default_root_dir=args.exp_dir
        )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()