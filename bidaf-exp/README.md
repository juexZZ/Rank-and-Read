# BiDAF Reader Experiments
My codes, experiment results and model checkpoints of our work *Bridging Neural Rankers and Machine Readers*

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

##### preprocess data

Run the script `preprocess.py` to preprocess the data, prepare for training and predicting:

```
python preprocess.py ../exp ../../train_v2.1.json --out_file train_data.json
```

By doing this, original training data `train_v2.1.json` will be loaded and preprocessed,  the preprocessed data will be store in the file: `../exp/preprocessed/train_data.json`. By default new vocabulary will be built at the same time, but you can skip this by setting argument `--renew_vocab` as False.

Changing the  argument to preprocess the development data:

```
python preprocess.py ../exp ../../dev_v2.1.json --out_file predict_data.json --renewvocab False
```

The preprocessed data will be stored in: `../exp/preprocessed/predict_data.json`.

Other arguments detail: see `preprocess.py` for more info.

##### training

1. To train the BiDAF baseline, run:

   ```
   python train_bidaf.py
   ```

   By doing this, arguments in `args.yaml` and configurations in `config.yaml` will be loaded.

   One should specify the arguments in `args.yaml` every time before training:

   ```yaml
   exp_folder: ../exp
   loss_record: bidaf_loss
   checkpoint: bidaf_epoch1_checkpoint
   data: /preprocessed/train_data.json
   force_restart: true
   word_rep: /data/disk4/private/zhangjuexiao/zlsh80826Marco/glove.840B.300d.txt
   cuda: true
   use_covariance: false
   ```

   

2. To train other models, run the script `train.py` with a  specific model name. For example:

   ```
   python train.py e2e_bidafcknrm
   ```

   will load in the arguments in `args.yaml` and configurations in `config.yaml`, and train an end-to-end BiDAF+ConvKNRM model.

   Other model names:

   `e2e_bidafknrm`: end-to-end BiDAF+K-NRM

   `multitask_bidafcknrm`: multitask BiDAF+ConvKNRM

   `multitask_bidafknrm`: multitask BiDAF+K-NRM

   `rar`: r-a-r multitask model by our implementation

   

##### predicting

Trained models' checkpoint can be found in `../exp` by default. 

To generate predicted answer spans, run the following commands:

1. For BiDAF baseline:

   ```
   python predict_bidaf.py
   ```

   Similar to training, arguments in `predict.yaml` will be loaded.

2. For BiDAF pipeline models:

   Still run `predict_bidaf.py`, but remember modify the file `predict.yaml`: set argument `pipeline ` as `true`, and specify `score_data` and `topk`.

   Check the file `predict.yaml` for more info.

3. For other models:

   Similar to training, run the script `predict.py` with specific model name. Remember to make sure the name matches the checkpoint in `predict.yaml`

   For example:

   ```
   python predict.py e2e_bidafcknrm
   ```

As the result, candidate answer files will be stored in the directory: `../exp/prediction/` by default.



