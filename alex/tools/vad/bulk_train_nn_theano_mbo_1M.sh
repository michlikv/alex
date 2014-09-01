#!/bin/sh
## Fail: python.sh
#

DEV='gpu'
EP='1000'
MF='1000000'
MFS='1M'

BS='30000'
BSS='30k'

LF=0
PF=0

HU=32

M='sg-rprop'
A='tanh'

export THEANO_FLAGS=mode=FAST_RUN,device=$DEV,floatX=float32
nohup  ./train_voip_nn_theano_sds_mfcc.py --max_epoch $EP --method $M --hact $A --batch_size $BS --max_frames $MF --hidden_units $HU --last_frames $LF --prev_frames $PF --mel_banks_only 1 > log.nnt_mf_$MFS.hu_$HU.lf_$LF.pf_$PF.mbo.bs_$BSS.$M.$A &


