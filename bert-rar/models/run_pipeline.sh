export MARCO_DIR=your marcoms dataset path

python run_pipeline.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --train_file $MARCO_DIR/train_v2.1.json \
  --predict_file $MARCO_DIR/dev_v2.1.json \
  --train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir your output dir
