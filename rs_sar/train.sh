# Train a model. Edit --cfg to point to the desired config under config_files/.
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg ./config_files/svnet_freqpkc.json \
    >train_svnet_freqpkc.log 2>&1
