# Rank-and-Read - bert reader

This is the repository of the codes of Rank-and-Read(bert reader) for MSMARCO QA task.

|Reader|Ranker|Bleu-1|Rouge-l|MRR@10|P@1 |
|:-----|:-----|:-----|:------|:-----|:---|
|Pipeline||||||
|BERT  |TFIDF |28.55 |25.58  |0.42  |0.26|
|BERT  |BM25  |31.07 |27.57  |0.46  |0.31|
|BERT  |KNRM  |35.51 |32.13  |0.53  |0.40|
|BERT  |CKNRM |39.14 |34.87  |0.57  |0.43|
|Multi-Task||||||
|BERT  |KNRM  |39.14 |35.91  |0.60  |0.55|
|BERT  |CKNRM |40.84 |36.59  |0.61  |0.55|
|End-to-End||||||
|BERT  |KNRM  |38.71 |35.29  |0.58  |0.50|
|BERT  |CKNRM |40.07 |36.27  |0.61  |0.57|

### Environment Requirement
- Python 3.6
- PyTorch 1.0

### Data Download & Preparation
To get MSMARCO QA dataset, see [MSMARCO QA](http://www.msmarco.org/dataset.aspx)

### Model
- To run our models, you need to install [pytorch pretrained bert](https://github.com/huggingface/pytorch-pretrained-BERT) first. Then add the code in models/modeling.py to the end of (pytorch pretrained bert path)/modeling.py that you just installed.

- When run the pipeline model, you will need to generate a ranking scores of passages of each query in MSMARCO dev set first, save it to a score file, and modify the dev score file in models/run_pipeline.py

- You need to modify:
    - the marcoms dataset & output path in \*.sh,
    - the pretrained bert/vocab.txt path in models/run_\*.py.

### training & prediction
```shell
./models/run_*.sh
```

### Evaluation
To evaluate the prediction file of bert, you need to transfer it to MSMARCO standard form. you can run evaluate/\*.py files for that, you need modify the prediction and output file path in evaluate/\*.py first.
```shell
python evaluate/*.py
```

### Contact
If you have any question, suggestions or bug reports, please email at zkt18@mails.tsinghua.edu.cn
