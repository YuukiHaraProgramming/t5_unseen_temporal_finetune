# T5 unseen temporal finetune

## Merge the TempLAMA datasets
```
RAW_DATA_DIR=/mnt/mint/hara/datasets/templama/preprocessed/multiple_answers_list
DATA_DIR=/mnt/mint/hara/datasets/templama/preprocessed/unseen/
$ python data_preprocess/merge_templama.py --data_dir $RAW_DATA_DIR --out_dir $DATA_DIR
```

## Process TempLAMA
- Output training data from all.json by removing data from years with no change in target.
- Split the non train data to validation data and test data.


## Fine-tune T5 as temporal model
```
DATA_DIR=/mnt/mint/hara/datasets/templama/preprocessed/unseen
EXP_DIR=/mnt/mint/hara/t5_temporal_finetune/unseen/07
python src/main.py --exp_dir $EXP_DIR --data_dir $DATA_DIR --batch_size 128 --model_name 't5-small' --lr 1e-3 --temporal_model
```

## Evaluate Fine-tuned T5
```
DATA_PATH=/mnt/mint/hara/datasets/templama/preprocessed/unseen/test.json
CKPT_PATH=/mnt/mint/hara/t5_temporal_finetune/unseen/06/best.ckpt
python evaluation.py --checkpoint_file $CKPT_PATH --test_dataset_path $DATA_PATH --model_name 't5-large'
```