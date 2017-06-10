#!/bin/bash

INPUT_FILE=$1
OUTPUT_FILE=$2

# Set this to where you extracted the downloaded file
#export DATA_PATH=/scratch/wellman_fluxg/lvnguyen/seq2seq/data
export DATA_PATH=./nmt_data/wmt_polarity

#export DEV_SOURCES=${DATA_PATH}/newstest2013.tok.bpe.32000.en
export DEV_SOURCES=${INPUT_FILE}

export MODEL_DIR=${TMPDIR:-./trained_models_full}/polarity_full
mkdir -p $MODEL_DIR

export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  > ${PRED_DIR}/${OUTPUT_FILE}
#predictions_polarity.txt
