import os
import random
import time
import pickle
import numpy as np
import argparse

import sys
sys.path.insert(0, os.path.abspath('./'))

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import s3dg
from datasets.coin.coin_video_loader import COIN_DataLoader
from utils.common_utils import log


import pdb



def get_args(
    description = \
        'Feature extraction for COIN using S3D model trained with MIL-NCE.'):
    
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--video_id_file', type=str,
                        default='annotations/COIN.json', 
                        help='pickle file that stores video ids and paths')
    parser.add_argument('--video_path', type=str,
                        default='', help='video dir')
    parser.add_argument('--word2vec_path', type=str, 
                        default='data/word2vec.pth', help='word2vec.pth path')
    
    parser.add_argument('--s3d_dict_path', type=str,
                        default='checkpoint/s3d_dict.npy', help='path to s3d_dict.npy')
    parser.add_argument('--s3d_howto100m_path', type=str,
                        default='checkpoint/s3d_howto100m.pth', help='path to s3d_howto100m.pth')
    parser.add_argument('--token_to_word_path', type=str,
                        default='data/dict.npy', help='path to token_to_word dict')
    
    parser.add_argument('--feat_save_root', type=str,
                        default='/export/home/data/coin/feats',
                        help='feature save dir')
    parser.add_argument('--log_root', type=str,
                        default='./log', help='log save dir')
    parser.add_argument('--log_filename', type=str,
                        default='feat_extract', help='log save file name')
     
    parser.add_argument('--n_display', type=int, default=10, 
                        help='Information display frequence')
    
    parser.add_argument('--fps', type=int, default=5, 
                        help='FPS')
    parser.add_argument('--num_frames', type=int, default=16, 
                        help='num frames')
    parser.add_argument('--clip_window', type=float, default=3.2,
                        help='assuming clip_window is a multiple of 3.2')
    parser.add_argument('--max_words', type=int, default=20,
                        help='max words')
    
    parser.add_argument('--video_size', type=int, default=224, 
                        help='video spatio size')
    parser.add_argument('--crop_only', type=int, default=1, 
                        help='crop only')
    parser.add_argument('--center_crop', type=int, default=0,
                        help='center crop')
    parser.add_argument('--random_flip', type=int, default=1,
                        help='random flip')
    
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='')
    parser.add_argument('--cudnn_benchmark', type=int, default=0,
                        help='')
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                        help='use pin_memory')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed. ')
    
    parser.add_argument('--early_stop', type=int, default=0, 
                        help='early stop or not')
    parser.add_argument('--early_stop_bid', type=int, default=0, 
                        help='early stop batch id')
    
    args = parser.parse_args()
    
    return args


def main():
    args = get_args()
    
    os.makedirs(args.feat_save_root, exist_ok=True)
    os.makedirs(args.log_root, exist_ok=True)
    
    if os.path.exists(os.path.join(args.log_root, args.log_filename + '.txt')):
        os.remove(os.path.join(args.log_root, args.log_filename + '.txt'))
    
    log("{}".format(args), args)
        
    assert args.video_path != ''
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    ngpus_per_node = torch.cuda.device_count()
    log('ngpus_per_node: {}'.format(ngpus_per_node), args)

    if args.num_workers == -1:
        args.num_workers = torch.get_num_threads() - 1
    log('num_workers: {}'.format(args.num_workers), args)
    
    
    main_worker(ngpus_per_node, args)

    


def main_worker(ngpus_per_node, args):
    
    # Instantiate the model
    model = s3dg.S3D(args.s3d_dict_path, 512)
    num_params = sum(param.numel() for param in model.parameters())
    log('model num. paramters: {}'.format(num_params), args)

    # Load the model weights
    model.load_state_dict(torch.load(args.s3d_howto100m_path))

    model = torch.nn.DataParallel(
        model, device_ids=[int(i) for i in range(ngpus_per_node)]).cuda()
    
    # Data loading code
    dataset = COIN_DataLoader(
        args=args,
        video_id_file=args.video_id_file,
        video_root=args.video_path,
        clip_window=args.clip_window,
        fps=args.fps,
        num_frames=args.num_frames,
        size=args.video_size,
        crop_only=args.crop_only,
        center_crop=args.center_crop,
        max_words=args.max_words,
        random_left_right_flip=args.random_flip,
        token_to_word_path=args.token_to_word_path
    )
    
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=dataset.custom_collate,
        pin_memory=args.pin_memory
    )

        
    if args.cudnn_benchmark:
        cudnn.benchmark = True
    
    # START feat extraction
    model = model.eval()
    
    start_time = time.time()
    feat_extract(data_loader, dataset, model, args)
    log("\n\nFeat extraction took {} s".format(round(time.time() - start_time, 2)), 
        args
    )
    
    
    
    

@torch.no_grad()
def feat_extract(data_loader, dataset, model, args):
    
    feat_extract_start_time = time.time()
    
    if os.path.exists(os.path.join(args.feat_save_root, 'problematic_videos.pickle')):
        with open(os.path.join(args.feat_save_root, 'problematic_videos.pickle'), 'rb') as f:
            problematic_videos = pickle.load(f)
    else:
        problematic_videos = dict()
    if os.path.exists(os.path.join(args.feat_save_root, 'corrupted_videos.pickle')):
        with open(os.path.join(args.feat_save_root, 'corrupted_videos.pickle'), 'rb') as f:
            corrupted_videos = pickle.load(f)
    else:
        corrupted_videos = set()
        
    
    batch_time = time.time()
    with torch.no_grad():
        for i_batch, sample_batch in enumerate(data_loader):
            data_time = time.time()

            video_id = sample_batch['video_id']  
            # video_id: a list of video ids
            vide_read_ok = sample_batch['vide_read_ok']
            # vide_read_ok: a list of bools
            """
            B: batch_size
            P: padded_max_clip_num (after padding, the num of clips of a video)
            S: subclip_num_of_each_clip (the num of subclips of each clip in a video)
            """
            clips = sample_batch['clips']
            # clips: (B, P, S, 3, 32, 224, 224) 
            # E.g., (2, 29, 3, 3, 32, 224, 224)
            start_times = sample_batch['start_times'] 
            # start_times: (B, P, S). 
            # E.g., (2, 29, 3)
            
            if False in vide_read_ok:  # some videos are corrupted or problematic
                video_id_ok = []
                for v_id in range(len(vide_read_ok)):
                    if vide_read_ok[v_id]:
                        video_id_ok.append(video_id[v_id])
                    else:
                        corrupted_videos.add(video_id[v_id])
                        with open(os.path.join(args.feat_save_root, 'corrupted_videos.pickle'), 'wb') as f:
                            pickle.dump(corrupted_videos, f)
                        
                video_id = video_id_ok
                clips = clips[vide_read_ok]
                start_times = start_times[vide_read_ok]
                
            if sum(vide_read_ok) > 0:
                try:
                    clips = clips.float().cuda(non_blocking=args.pin_memory)
                    clips = clips / 255.0

                    # pdb.set_trace()
                    video_output = model(
                        **dict(video=clips.flatten(0,1).flatten(0,1),  # (B', 3, 32, 224, 224) where B'=BxPxS
                        text=None,
                        mode='video', mixed5c=False, raw_text=False))
                    
                    video_embedding = video_output['video_embedding']  # (B', 512)
                    
                    # pdb.set_trace()
                    
                    last_none_padding_index = dataset.find_last_none_padding_index(start_times)

                    ### Save features
                    B, P, S = start_times.shape
                    video_embed_np = video_embedding.view(B,P,S,-1).cpu().numpy() 
                    for v_id in range(len(video_id)):
                        v_sid = video_id[v_id]

                        os.makedirs(os.path.join(args.feat_save_root, v_sid), exist_ok=True)

                        np.save(
                            os.path.join(args.feat_save_root, v_sid, 'video.npy'),
                            video_embed_np[v_id][:last_none_padding_index[v_id]]  # (actual_P, S, 512)
                        )
                        np.save(
                            os.path.join(args.feat_save_root, v_sid, 'segment_time.npy'),
                            start_times[v_id][:last_none_padding_index[v_id]]  # (actual_P, S)
                        )
                        with open(os.path.join(args.feat_save_root, v_sid, 'status.txt'), 'w') as f:
                            f.write('succeeded!\n'.format(i_batch, len(data_loader)))

                except Exception as e:   # E.g., RuntimeError: CUDA OOM
                    message = "Error message: {}".format(e)
                    log("{}".format(message), args)

                    for v_id in range(len(video_id)):
                        problematic_videos[video_id[v_id]] = message
                        
                    with open(os.path.join(args.feat_save_root, 'problematic_videos.pickle'), 'wb') as f:
                        pickle.dump(problematic_videos, f)
                        
            torch.cuda.empty_cache()

            if (i_batch + 1) % args.n_display == 0:
                log("{} / {}  Time of the Batch --- Data: {} s\tData & Processing: {} s\tSo Far: {} s".format(
                    i_batch, len(data_loader),
                    round(data_time - batch_time, 2),
                    round(time.time() - batch_time, 2),
                    round(time.time() - feat_extract_start_time, 2)) +
                    "\tNum Corrupted: {}\tNum Problematic: {}".format(
                    len(corrupted_videos),
                    len(problematic_videos)
                ), args)
                

            batch_time = time.time()

            if args.early_stop and i_batch == args.early_stop_bid:
                break
                
#     log("{} problematic videos.".format(len(problematic_videos)), args)  
#     if len(problematic_videos) > 0:            
#         with open(os.path.join(args.feat_save_root, 'problematic_videos.pickle'), 'wb') as f:
#             pickle.dump(problematic_videos, f)
#         log("Saved problematic videos in {}".format(
#             os.path.join(args.feat_save_root, 'problematic_videos.pickle')), args)



 
              
              
if __name__ == "__main__":
    main()
