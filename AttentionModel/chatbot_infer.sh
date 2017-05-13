# Define variables #
export TEST_INPUT=$PWD/act_test.src
export MODEL_DIR=$PWD/S2SModels
export PRED_DIR=$MODEL_DIR/pred
mkdir -p ${PRED_DIR}

# Tokenize and bpe input #
./tokenizer.perl < $TEST_INPUT > act_test.src.tok
export TEST_SOURCES=$PWD/act_test.src.tok

# Inference #
python3 -m bin.infer \
  --tasks "
    - class: DecodeText
      params:
        unk_replace: True" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
        source_files:
          - $TEST_SOURCES" \
    > ${PRED_DIR}/predictions.txt

