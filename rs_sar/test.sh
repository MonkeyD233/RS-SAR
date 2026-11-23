# Test a pretrained model. Edit --cfg and --model-path first.
CUDA_VISIBLE_DEVICES=0 python -u test.py \
    --cfg ./config_files/svnet_freqpkc.json \
    --model-path /path/to/model.pt
