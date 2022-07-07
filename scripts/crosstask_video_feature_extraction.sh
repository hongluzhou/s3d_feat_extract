time python main/crosstask_video_feature_extraction.py --n_display=1 \
       --batch_size=1 \
       --num_workers=-1 --cudnn_benchmark=0 --pin_memory \
       --fps=10 \
       --num_frames=32 \
       --clip_window=9.6 \
       --max_words=16 \
       --video_size=224 \
       --crop_only=0 \
       --center_crop=1 \
       --random_flip=0 \
       --early_stop=0 \
       --s3d_dict_path=/export/home/code/ginst/howto100m/ginst-HowTo100M-Feat_Extract/checkpoint/s3d_dict.npy \
       --s3d_howto100m_path=/export/home/code/ginst/howto100m/ginst-HowTo100M-Feat_Extract/checkpoint/s3d_howto100m.pth \
       --token_to_word_path=/export/home/code/ginst/howto100m/ginst-HowTo100M-Feat_Extract/data/dict.npy \
       --video_id_file=/export/einstein-vision/multimodal_video/datasets/CrossTask/video_ids_successfully_downloaded.pickle \
       --video_path=/export/einstein-vision/multimodal_video/datasets/CrossTask/videos \
       --feat_save_root=/export/home/data/crosstask/debug_feats \
       --log_root=./logs \
       --log_filename=crosstask_video_feature_extraction
       


# --video_id_csv=/export/home/code/ginst/howto100m/ginst-HowTo100M-Feat_Extract/train.csv \
# --video_id_csv=/export/home/code/ginst/howto100m/MIL-NCE_HowTo100M/csv/howto100m_videos.csv \
