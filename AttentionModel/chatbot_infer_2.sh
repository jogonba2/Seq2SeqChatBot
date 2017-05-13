python3 -m bin.infer \
  --tasks "
    - class: DecodeText
      params:
        unk_replace: True" \
  --model_dir ./S2SModels/  \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
        source_files:
          - ./795033" \
    > ./S2SModels/795033.preds



# Define variables #
export TEST_INPUT=$PWD/act_test.src
export MODEL_DIR=$PWD/S2SModels
export PRED_DIR=$MODEL_DIR/pred
mkdir -p ${PRED_DIR}

# Inference #
#python3 -m bin.infer \
#  --tasks "
#    - class: DecodeText
#      params:
#        unk_replace: True" \
#  --model_dir $MODEL_DIR \
#  --input_pipeline "
#    class: ParallelTextInputPipeline
#    params:
#        source_files:
#          - $TEST_SOURCES" \
#    > ${PRED_DIR}/predictions.txt

