export -x DISPLAY :0.0
set -x DISPLAY :0.0

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64

# Tokenize #
./tokenizer.perl < ./x_train.txt > train.src.tok
./tokenizer.perl < ./x_test.txt > test.src.tok
./tokenizer.perl < ./y_train.txt > train.trg.tok
./tokenizer.perl < ./y_test.txt > test.trg.tok


# Get train vocab #
python3 generate_vocab.py < train.src.tok > vocab.src  
python3 generate_vocab.py < train.trg.tok > vocab.trg 

# Define variables #
export VOCAB_SOURCE=$PWD/vocab.src
export VOCAB_TARGET=$PWD/vocab.trg
export TRAIN_SOURCES=$PWD/train.src.tok
export TRAIN_TARGETS=$PWD/train.trg.tok
export DEV_SOURCES=$PWD/test.src.tok
export DEV_TARGETS=$PWD/test.trg.tok
export DEV_TARGETS_REF=$DEV_TARGETS
export TRAIN_STEPS=100000

mkdir ./S2SModels
export MODEL_DIR=./S2SModels

# Training #
python3 -m bin.train \
  --config_paths "
    ./chatbot_complex_config.yml,
    ./text_metrics_bpe.yml,
    ./train_seq2seq.yml" \
  --model_params "
    vocab_source: $VOCAB_SOURCE
    vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
       - $TRAIN_SOURCES
      target_files:
       - $TRAIN_TARGETS"\
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
      source_files:
       - $DEV_SOURCES
      target_files:
       - $DEV_TARGETS"\
  --batch_size 256 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR
