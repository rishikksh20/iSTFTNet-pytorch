python3 train.py --config config_v1.json \
    --input_wavs_dir /home/ubuntu/dev/dataset/pino_it/wavs_32kHz \
    --input_ds_dir /home/ubuntu/dev/dataset/pino_it/wavs_16kHz \
    --input_training_file /home/ubuntu/dev/dataset/pino_it/filelist/train.txt \
    --input_validation_file /home/ubuntu/dev/dataset/pino_it/filelist/valid.txt \
    --checkpoint_path /home/ubuntu/dev/dataset/pino_it/pino_ft_hifitts_16_ups_32 \
    --checkpoint_interval 10000 \
    --fine_tuning True \
    --training_epochs 20000