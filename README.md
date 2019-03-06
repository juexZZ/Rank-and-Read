# Rank-and-Read
My codes, exp results and model checkpoints of the bridging neural rankers and machine readers

## models/checkpoints
checkpoints of all my trained models are here:

https://drive.google.com/file/d/1CN4_DO-NTXApnCgSMIVEFWI8LcfDeXzh/view?usp=sharing

By downloading and unzip this file, you should have all the models' checkpoints that can give the same results in our paper through inference/prediction.

The results can also be download here:

https://drive.google.com/drive/folders/1GiRpmHyT1_njzQp5UwlmbRba7DhHPte6?usp=sharing

Both reading comprehension results(candidate answer spans in `readings.zip`) and ranker results(ranking scores for each passage in `rankings.zip`) are available. 

## file structures

`models` 

├── `MyRanker.py` two neural ranker modules:  K-NRM, ConvKNRM

├──`bidaf.py` BiDAF baseline, implemented by MSMARCO

├──`e2e_bidafcknrm.py` end-to-end model BiDAF+ConvKNRM

├──`e2e_bidafknrm.py`end-to-end model BiDAF+K-NRM

├──`embeddings.py` update/static word embedding classes and character embedding class

├──`highway.py` highway layer class, used by BiDAF reader

├──`multitask_bidafcknrm.py` multitask model BiDAF+ConvKNRM

├──`multitask_bidafknrm.py`multitask model BiDAF+KNRM

└──`rar.py` implementation of r-a-r model

`util`

├──`text_input.py` text preprocess functions

└──`utils.py`  ranker utilities

`Mydataset.py` dataloader and relevant functions(adopted from MSMARCO `dataset.py`)

`args.yaml` arguments for training

`checkpointing.py` checkpointing functions(adopted from MSMARCO `checkpointing.py`)

`config.yaml` configurations for model, training and predicting

`config_rar.yaml` configurations for r-a-r, training and predicting

`predict.py` run to predict(infer) all models except BiDAF baseline

`predict.yaml` arguments for predicting

`predict_bidaf.py` run to predict BiDAF baseline

`preprocess.py` run to preprocess dataset

`train.py` run to train all models except BiDAF baseline

`train_bidaf.py` run to train BiDAF baseline

## how to run

