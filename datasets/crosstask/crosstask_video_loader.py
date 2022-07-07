import os
import numpy as np
import pandas as pd
import random
import ffmpeg
import time
import re
import math
import pickle
import pdb
import glob

from utils.common_utils import log

import torch as th
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence



class CrossTask_DataLoader(Dataset):
    """CrossTask Video loader."""

    def __init__(
            self,
            args,
            video_id_file,
            video_root='',
            clip_window=4.0,
            fps=16,
            num_frames=16,
            size=224,
            crop_only=False,
            center_crop=True,
            max_words=20,
            random_left_right_flip=False,
            token_to_word_path='data/dict.npy'
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        
        self.args = args
        self.video_root = video_root
        self.clip_window = clip_window
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.num_sec =  self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.max_words = max_words
        self.token_to_word = np.load(token_to_word_path)
        self.word_to_token = {}
        for i, t in enumerate(self.token_to_word):
            self.word_to_token[t] = i + 1  # plus 1 because 0 is used as padding
        self.random_flip = random_left_right_flip
        
        video_ids = pickle.load(open(video_id_file, "rb"))
        
        # Ignore videos whose features were extracted
        self.video_ids = []
        if os.path.exists(os.path.join(args.feat_save_root, 'corrupted_videos.pickle')):
            with open(os.path.join(args.feat_save_root, 'corrupted_videos.pickle'), 'rb') as f:
                corrupted_videos = pickle.load(f)
        else:
            corrupted_videos = set()
        for video_id in video_ids:
            if os.path.exists(os.path.join(args.feat_save_root, video_id, 'status.txt')):
                continue  # features were extracted
            else:
                if video_id not in corrupted_videos:
                    self.video_ids.append(video_id)
                
                
        ### DEBUG certain videos                    
        # self.video_ids = ['L6YDBFbnpiw']
        ### DEBUG certain videos   
        
        
        log('{} samples...'.format(self.__len__()), self.args)  
        # 4,577 samples... 
            
            
        # pdb.set_trace()
        # self.__getitem__(0)
        # pdb.set_trace()
        # self.__getitem__(1)
        # pdb.set_trace()
        
        

    def __len__(self):
        return len(self.video_ids)

    def _get_video(self, video_path, start_seek):
        
        vide_read_ok = True
        
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec + 0.1)
            .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        if self.random_flip and random.uniform(0, 1) > 0.5:
            cmd = cmd.hflip()
        try:
            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
        except ffmpeg._run.Error:
            vide_read_ok = False
            log("Video corrupted! Path: {}".format(video_path), self.args)
            return th.zeros(
                (3, self.num_frames, self.size, self.size), 
                dtype=th.uint8), vide_read_ok
        
        
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        # print(video.shape)
        ### DEBUG
        # from PIL import Image
        # for f_idx in range(len(video)):
        #     im = Image.fromarray(video[f_idx])
        #     im.save("tem/{}-startseek{}_frame{}.jpeg".format(
        #         video_path.split('/')[-1].split('.')[0], start_seek, f_idx))
        # pdb.set_trace()
        ### DEBUG
        video = th.from_numpy(video.copy())
        # pdb.set_trace()
        video = video.permute(3, 0, 1, 2)
        # print(video_path, start_seek, video.shape)
        if video.shape[1] < self.num_frames:
            zeros = th.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        # pdb.set_trace()
        return video[:, :self.num_frames], vide_read_ok

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words, dtype=th.long)

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))
    
    def _possibly_reach_end_of_video(self, clip):
        if th.sum(clip):
            return False
        else:  # if the values are all 0s, means it might be the end
            return True
        
    def _reach_end_of_video(self, possibly_end, thresh=10):
        if len(possibly_end) < thresh:
            return False
        else:
            if sum(possibly_end[-thresh:]) == thresh:
                return True
            else:
                return False
            
    def _get_caption(self, cap, start):
        """ pparam: 
        - start: the start time of the (sub)clip
        """
        res = []  # store captions of the (sub)clip that starts at start
        start_ts = cap['start'].values
        end = start + self.num_sec
        for i in range(len(start_ts)):  # loop over start time of captions
            if start_ts[i] > end:  # start time of caption > end time of clip
                break
            else: 
                if i+1 < len(start_ts) and start_ts[i+1] - 0.01 < start:  # end time of caption < start time of clip
                    continue
                else:
                    if start_ts[i] + self.num_sec >= start:
                        # if start time of caption <= end time of clip
                        # AND end time of caption >= start time of clip
                        # then, we keep the caption
                        res.append(cap['text'].values[i])
        return " ".join(s for s in res)
                    
        
    
    def _get_video_segments(self, video_path):
        """ Note: 
        - assuming self.clip_window is a multiple of self.num_sec 
        - self.num_sec is expected to be 3.2 (https://arxiv.org/abs/1912.06430)
        """
        
        ### load clips of video
        start = 0
        end = start + self.clip_window
        
        clips, clip = [], []
        times, time = [], []  # stores the start time of each subclip in each clip
        
        possibly_end = []
        # count = 0
        thresh = 100  # consider video ends if (self.num_sec * thresh) seconds of empty frames
        while not self._reach_end_of_video(possibly_end, thresh=thresh):
            # count += 1
            subclip, vide_read_ok = self._get_video(video_path, start)
            if not vide_read_ok:
                clips_stacked = th.zeros(
                    (0, round(self.clip_window/self.num_sec), 3, self.num_frames, self.size, self.size), dtype=th.uint8)
                times_stacked = th.zeros((0, round(self.clip_window/self.num_sec)))
                return vide_read_ok, clips_stacked, times_stacked
                
            if start <= end:
                clip.append(subclip)
                time.append(start)
                
                if math.isclose(start + self.num_sec, end):
                    end += self.clip_window
                    clips.append(clip)
                    clip = []
                    times.append(time)
                    time = []
                    
            start += self.num_sec
                
            if self._possibly_reach_end_of_video(subclip):
                possibly_end.append(1)
            else:
                possibly_end.append(0)
                
            # print(count, start, end, possibly_end[-1], vide_read_ok, len(clip), len(clips))
            # pdb.set_trace()
        
        thresh -= len(clip)
        while thresh >= self.clip_window / self.num_sec:
            thresh -= self.clip_window / self.num_sec
            clips.pop()
            times.pop()
            
        # == Obtained clips, times.
        # == All of them are list of lists.
            
        clips_stacked = []
        for clip_idx in range(len(clips)):
            clips_stacked.append(th.stack(clips[clip_idx]))
        
        # pdb.set_trace()
        if len(clips_stacked) > 0:
            clips_stacked = th.stack(clips_stacked)  # torch.Size([clip_num, subclip_num_of_each_clip, 3, 32, 224, 224])
            times_stacked = th.tensor(times)  # torch.Size([clip_num, subclip_num_of_each_clip])

            return vide_read_ok, clips_stacked, times_stacked
        else:
            vide_read_ok = False
            clips_stacked = th.zeros(
                (0, round(self.clip_window/self.num_sec), 3, self.num_frames, self.size, self.size), dtype=th.uint8)
            times_stacked = th.zeros((0, round(self.clip_window/self.num_sec)))
            return vide_read_ok, clips_stacked, times_stacked
        
        
    @staticmethod  
    def custom_collate(batch):
        # https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3
        video_id = [data['video_id'] for data in batch]
        vide_read_ok = [data['vide_read_ok'] for data in batch]
        
        clips = pad_sequence(
            [data['clips'] for data in batch], batch_first=True)
        start_times = pad_sequence(
            [data['start_times'] for data in batch], batch_first=True)
        
        return {
            'video_id': video_id, 
            'vide_read_ok': vide_read_ok,
            'clips': clips,
            'start_times': start_times
        }
    
    @staticmethod  
    def find_last_none_padding_index(start_times):
        """
        start_times: (B, P, S)
        """
        res = []
        for b_idx in range(len(start_times)):
            for p_idx in range(len(start_times[b_idx])):
                if th.sum(start_times[b_idx][p_idx:]):
                    continue
                else:
                    res.append(p_idx)  # start_times[b_idx][:p_idx] are non-padding
                    break
            if p_idx == len(start_times[b_idx]) - 1: 
                res.append(p_idx + 1)
                
        return res
    

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        paths_this_video = glob.glob(os.path.join(self.video_root, "*", "{}.*".format(video_id)))
        video_path = paths_this_video[0]
        
        vide_read_ok, clips, start_times = self._get_video_segments(video_path)
        
        return {'video_id': video_id, 
                'vide_read_ok': vide_read_ok,
                'clips': clips, 
                'start_times': start_times
               }

    