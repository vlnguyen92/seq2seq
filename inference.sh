#!/bin/bash

# Set this to where you extracted the downloaded file
#export DATA_PATH=/scratch/wellman_fluxg/lvnguyen/seq2seq/data
export DATA_PATH=./nmt_data/wmt_polarity

export VOCAB_SOURCE=${DATA_PATH}/vocab.bpe.32000
export VOCAB_TARGET=${DATA_PATH}/vocab.bpe.32000
export TRAIN_SOURCES=${DATA_PATH}/train.tok.clean.bpe.32000.en
export TRAIN_TARGETS=${DATA_PATH}/train.tok.clean.bpe.32000.en
#export DEV_SOURCES=${DATA_PATH}/newstest2013.tok.bpe.32000.en
export DEV_SOURCES=${DATA_PATH}/test.tok.clean.bpe.32000.en
export DEV_TARGETS=${DATA_PATH}/test.tok.clean.bpe.32000.en

export DEV_TARGETS_REF=${DATA_PATH}/test.tok.en

export MODEL_DIR=${TMPDIR:-./trained_models_full}/polarity_adv
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
  > ${PRED_DIR}/predictions_polarity.txt
