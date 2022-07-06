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
from datasets.wikihow.wikihow_step_loader import WikiHow_Step_DataLoader
from utils.common_utils import log

import pdb



def get_args(
    description = \
        'Feature extraction for WikiHow steps using S3D model trained with MIL-NCE.'):
    
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--word2vec_path', type=str, 
                        default='data/word2vec.pth', help='word2vec.pth path')
    parser.add_argument('--max_words', type=int, default=20,
                        help='max words')
    parser.add_argument('--wikihow_subset', type=int, default=1,
                        help='whether it is the subset or the full set')
     
    parser.add_argument('--s3d_dict_path', type=str,
                        default='checkpoint/s3d_dict.npy', help='path to s3d_dict.npy')
    parser.add_argument('--s3d_howto100m_path', type=str,
                        default='checkpoint/s3d_howto100m.pth', help='path to s3d_howto100m.pth')
    parser.add_argument('--token_to_word_path', type=str,
                        default='data/dict.npy', help='path to token_to_word dict')
    
    parser.add_argument('--feat_save_root', type=str,
                        default='/export/home/data/wikihow/wikihow_subset/feats',
                        help='feature save dir')
    parser.add_argument('--log_root', type=str,
                        default='./log', help='log save dir')
    parser.add_argument('--log_filename', type=str,
                        default='feat_extract', help='log save file name')
     
    parser.add_argument('--n_display', type=int, default=10, 
                        help='Information display frequence')
    
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='if -1, will automatically get all workers')
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
    wikihow_step_dataset = WikiHow_Step_DataLoader(
        args=args,
        wikihow_subset=args.wikihow_subset,
        max_words=args.max_words,
        token_to_word_path=args.token_to_word_path
    )
    
    
    data_loader = DataLoader(
        wikihow_step_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

        
    if args.cudnn_benchmark:
        cudnn.benchmark = True
    
    # START feat extraction
    model = model.eval()
    
    start_time = time.time()
    feat_extract(data_loader, wikihow_step_dataset, model, args)
    log("\n\nFeat extraction took {} s".format(round(time.time() - start_time, 2)), 
        args
    )
    
    
    
    

@torch.no_grad()
def feat_extract(data_loader, dataset, model, args):
    
    feat_extract_start_time = time.time()
    step_embeddings = np.zeros((len(dataset), 512))
    step_preprocessed = dict()
    problematic_steps = dict()
    
    batch_time = time.time()
    with torch.no_grad():
        for i_batch, sample_batch in enumerate(data_loader):
            data_time = time.time()

            step_id = sample_batch['step_id'].cpu().numpy() 
            step_sentence_preprocessed = sample_batch['step_sentence_preprocessed']
            step_tokens = sample_batch['step_tokens']
            
            try:
                step_tokens = step_tokens.cuda(non_blocking=args.pin_memory)

                text_output  = model(
                    **dict(video=None, 
                    text=step_tokens,    # (B, 16)
                    mode='text', mixed5c=False, raw_text=False))

                text_embedding = text_output['text_embedding']  # (B', 512)


                ### Save features
                text_embed_np = text_embedding.cpu().numpy() 
                for s_id in range(len(step_id)):
                    step_embeddings[step_id[s_id]] = text_embed_np[s_id]
                    step_preprocessed[step_id[s_id]] = step_sentence_preprocessed[s_id]
                
            except Exception as e:   # E.g., RuntimeError: CUDA OOM
                message = "Batch ID: {} Error message: {}".format(i_batch, e)
                log("{}".format(message), args)
                # pdb.set_trace()
               
                for s_id in range(len(step_id)):
                    problematic_steps[step_id[s_id]] = message
                

            torch.cuda.empty_cache()

            if (i_batch + 1) % args.n_display == 0:
                log("{} / {}  Time of the Batch --- Data: {} s\t\tData & Processing: {} s\t\t\tSo Far: {} s".format(
                    i_batch, len(data_loader),
                    round(data_time - batch_time, 2),
                    round(time.time() - batch_time, 2),
                    round(time.time() - feat_extract_start_time, 2)
                ), args)


            batch_time = time.time()

            if args.early_stop and i_batch == args.early_stop_bid:
                break
                
    with open(os.path.join(args.feat_save_root, 'step_embeddings.pickle'), 'wb') as f:
            pickle.dump(step_embeddings, f)
    log("Saved {}".format(
            os.path.join(args.feat_save_root, 'step_embeddings.pickle')), args)

    with open(os.path.join(args.feat_save_root, 'step_preprocessed.pickle'), 'wb') as f:
            pickle.dump(step_preprocessed, f)
    log("Saved {}".format(
            os.path.join(args.feat_save_root, 'step_preprocessed.pickle')), args)

                
    log("{} problematic steps.".format(len(problematic_steps)), args)
    if len(problematic_steps) > 0:
        with open(os.path.join(args.feat_save_root, 'problematic_steps.pickle'), 'wb') as f:
            pickle.dump(problematic_steps, f)

        log("Saved problematic steps in {}".format(
            os.path.join(args.feat_save_root, 'problematic_steps.pickle')), args)


              
              
if __name__ == "__main__":
    main()
