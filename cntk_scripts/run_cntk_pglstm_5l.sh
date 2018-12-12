#!/bin/bash

# Copyright 2012-2013 Karel Vesely, Daniel Povey
# 	        2017 Wei-Ning Hsu, Yu Zhang
# Apache 2.0

# This script trains and evaluate PGLSTM models. There is no
# In this recipe, CNTK directly read Kaldi features and labels,
# which makes the whole pipline much simpler.

. ./cmd.sh
. ./path.sh

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will 
                 #return non-zero return code
set -u           #Fail on an undefined variable

# experiment directory
expdir=exp_cntk/pglstm_5l_fbank
trainlog=train_cntk

# gmm model and labels
gmm_src=exp/tri5a
graph_src=${gmm_src}/graph
ali_src=${gmm_src}_ali

# features
train_src=data/train_fbank
test_src=data/dev_fbank

# cntk config and ndl file
config=cntk_conf/CNTK_lstm.config
ndl=cntk_conf/ndl_factory/pglstm_5l.ndl

# path to cntk binary
cn_gpu=cntk

# The device number to run the training
DeviceNumber=0

# decoding options
acwt=0.12
njdec=40
scoring="--min-lmwt 5 --max-lmwt 19"

stage=0
. utils/parse_options.sh || exit 1;

labelDim=$(($(cat ${ali_src}/final.occs | wc -w)-2))
baseFeatDim=$(feat-to-dim scp:${train_src}/feats.scp - | cat -)
# This effectively delays the output label by 5 frames, so that the LSTM sees 5 future frames.
featDim=$((baseFeatDim*11))
rowSliceStart=$((baseFeatDim*10))
mkdir -p $expdir

if [ ! -d ${train_src}_tr90 ] ; then
    echo "Training and validation sets not found. Generating..."
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 ${train_src} ${train_src}_tr90 ${train_src}_cv10
    echo "done."
fi

feats_tr="scp:${train_src}_tr90/feats.scp"
feats_cv="scp:${train_src}_cv10/feats.scp"
labels_tr="ark:ali-to-pdf $ali_src/final.mdl \"ark:gunzip -c $ali_src/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"

if [ $stage -le 0 ] ; then
    (feat-to-len "$feats_tr" ark,t:- > $expdir/cntk_train.counts) || exit 1;
    echo "$feats_tr" > $expdir/cntk_train.feats
    echo "$labels_tr" > $expdir/cntk_train.labels

    (feat-to-len "$feats_cv" ark,t:- > $expdir/cntk_valid.counts) || exit 1;
    echo "$feats_cv" > $expdir/cntk_valid.feats
    echo "$labels_tr" > $expdir/cntk_valid.labels

    for (( c=0; c<labelDim; c++)) ; do
        echo $c
    done >$expdir/cntk_label.mapping
fi

if [ $stage -le 1 ] ; then
    ### setup the configuration files for training CNTK models ###
    cp cntk_conf/default_macros.ndl $expdir/
    cp $config $expdir/CNTK2.config
    cp $ndl $expdir/nn.ndl
    ndlfile=$expdir/nn.ndl

    tee $expdir/Base.config <<EOF
ExpDir=$expdir
logFile=${trainlog}
modelName=cntk.nn

verbosity=0
labelDim=${labelDim}
featDim=${baseFeatDim}
labelMapping=${expdir}/cntk_label.mapping
featureTransform=NO_FEATURE_TRANSFORM

inputCounts=${expdir}/cntk_train.counts
inputFeats=${expdir}/cntk_train.feats
inputLabels=${expdir}/cntk_train.labels

cvInputCounts=${expdir}/cntk_valid.counts
cvInputFeats=${expdir}/cntk_valid.feats
cvInputLabels=${expdir}/cntk_valid.labels
EOF

    ## training command ##
    $cuda_cmd $expdir/log/cmdtrain.log \
        $cn_gpu configFile=${expdir}/Base.config configFile=${expdir}/CNTK2.config \
        DeviceNumber=$DeviceNumber action=TrainLSTM ndlfile=$ndlfile \
        FeatDim=$featDim baseFeatDim=$baseFeatDim RowSliceStart=$rowSliceStart

    echo "$0 successfuly finished training... $expdir"

fi

if [ $stage -le 2 ]; then
    config_write=cntk_conf/CNTK2_write.config
    cnmodel=${expdir}/cntk.nn
    action=write
    dec_stage=0
    cp ${ali_src}/final.mdl ${expdir}
    cntk_string="$cn_gpu configFile=$config_write verbosity=0 DeviceNumber=-1"
    cntk_string="$cntk_string modelName=$cnmodel labelDim=$labelDim featDim=$featDim"
    cntk_string="$cntk_string action=$action ExpDir=$expdir"

    # run decoding script
    cntk_scripts/cntk_decode.sh --nj $njdec --cmd $decode_cmd --acwt $acwt \
        --scoring-opts "$scoring" --stage $dec_stage \
        $graph_src $test_src $expdir/decode_$(basename $test_src) "$cntk_string" || exit 1;

    echo "$0 successfuly finished decoding... $expdir"
fi

exit 0
