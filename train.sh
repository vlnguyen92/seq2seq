#!/bin/bash

MACHINE=$1

if [ "$MACHINE" == "flux" ]; then
    echo "Running on flux"
    ROOT_DIR=/scratch/wellman_fluxg/lvnguyen/seq2seq
else
    echo "Running on conflux"
    ROOT_DIR=.
fi

# Set this to where you extracted the downloaded file
export DATA_PATH=$ROOT_DIR/nmt_data/movie_reviews

export VOCAB_SOURCE=${DATA_PATH}/vocab.bpe.32000
export VOCAB_TARGET=${DATA_PATH}/vocab.bpe.32000
export TRAIN_SOURCES=${DATA_PATH}/train.tok.clean.bpe.32000.en
export TRAIN_TARGETS=${DATA_PATH}/train.tok.clean.bpe.32000.en
export DEV_SOURCES=${DATA_PATH}/test.tok.bpe.32000.en
export DEV_TARGETS=${DATA_PATH}/test.tok.bpe.32000.en

export DEV_TARGETS_REF=${DATA_PATH}/test.tok.en
export TRAIN_STEPS=100000

export MODEL_DIR=${TMPDIR:-$ROOT_DIR/trained_models}/polarity
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 128\
  --train_steps $TRAIN_STEPS \
  --eval_every_n_steps 100000 \
  --output_dir $MODEL_DIR
