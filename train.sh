python3 train.py --config config_v3.json \
    --input_wavs_dir /home/ubuntu/dev/data/LJSpeech-1.1/wavs_16kHz \
    --input_training_file LJSpeech-1.1/training.txt \
    --input_validation_file LJSpeech-1.1/validation.txt \
    --checkpoint_path /home/ubuntu/dev/checkpoints/istft_ljspeech_16khz_v3 \
    --checkpoint_interval 10000