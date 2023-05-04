#!/bin/bash

# python3 src/train/train.py --train-paths irmastrain:data/irmas/train --val-paths irmastest:data/irmas/test --backbone-after layer.11 --head-after backbone.classifier --model AST --audio-transform AST --finetune-head-epochs 2 --batch-size 3 --num-workers 6 --quick --epochs 4 &&\
# python3 src/train/train.py --train-paths irmastrain:data/irmas/train --val-paths irmastest:data/irmas/test --audio-transform WAV2VEC --model WAV2VEC  --backbone-after layers.9 --head-after deep_head --finetune-head-epochs 2 --batch-size 3 --quick --epochs 4 &&\
python3 src/train/train.py --train-paths irmastrain:data/irmas/train --val-paths irmastest:data/irmas/test --backbone-after backbone.features.7.2 --head-after backbone.classifier --model EFFICIENT_NET_V2_S --audio-transform MEL_SPECTROGRAM --finetune-head-epochs 1 --batch-size 3 --accumulate_grad_batches 4 --quick --epochs 4 --loss-function FOCAL_LOSS --add-instrument-loss 0.3 &&\
python3 src/train/train.py --train-paths irmastrain:data/irmas/train --val-paths irmastest:data/irmas/test --backbone-after backbone.features.7.2 --head-after backbone.classifier --model EFFICIENT_NET_V2_S --audio-transform MEL_SPECTROGRAM --finetune-head-epochs 1 --batch-size 3 --accumulate_grad_batches 4 --quick --epochs 4 --loss-function FOCAL_LOSS &&\
python3 src/train/train.py --train-paths irmastrain:data/irmas/train --val-paths irmastest:data/irmas/test --backbone-after backbone.features.7.2 --head-after backbone.classifier --model EFFICIENT_NET_V2_S --audio-transform MEL_SPECTROGRAM --finetune-head-epochs 1 --batch-size 3 --accumulate_grad_batches 4 --quick --epochs 4 --loss-function FOCAL_LOSS --loss-function-kwargs gamma=1 &&\
python3 src/train/train.py --train-paths irmastrain:data/irmas/train --val-paths irmastest:data/irmas/test --backbone-after backbone.features.7.2 --head-after backbone.classifier --model EFFICIENT_NET_V2_S --audio-transform MEL_SPECTROGRAM --finetune-head-epochs 1 --batch-size 3 --accumulate_grad_batches 4 --quick --epochs 4 &&\
python3 src/train/train.py --train-paths irmastrain:data/irmas/train --val-paths irmastest:data/irmas/test --backbone-after backbone.fc --head-after backbone.fc --model RESNEXT50_32X4D --audio-transform MEL_SPECTROGRAM --finetune-head-epochs 1 --batch-size 3 --accumulate_grad_batches 4 --quick --epochs 4 &&\
python3 src/train/train.py --train-paths irmastrain:data/irmas/train --val-paths irmastest:data/irmas/test --backbone-after backbone.features.7.2 --head-after backbone.classifier --model CONVNEXT_TINY --audio-transform MEL_SPECTROGRAM --finetune-head-epochs 1 --batch-size 3 --accumulate_grad_batches 4 --quick --epochs 4 &&\
python3 src/train/train.py --train-paths irmastrain:data/irmas/train --val-paths irmastest:data/irmas/test --backbone-after backbone.avgpool --head-after backbone.classifier --model MOBILENET_V3_LARGE --audio-transform MEL_SPECTROGRAM --finetune-head-epochs 1 --batch-size 3 --accumulate_grad_batches 4 --quick --epochs 4
