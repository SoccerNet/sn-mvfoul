from torch.utils.data import Dataset

import numpy as np
from random import random
import os
import time
from tqdm import tqdm
import torch
import logging
import json
import random
from data_loader import label2vectormerge, clips2vectormerge
from torchvision.io.video import read_video, write_video
import time
from torchvision.models.video import MViT_V2_S_Weights


class MultiViewDataset(Dataset):
    def __init__(self, path, start, end, fps, split, types, num_views, action=False, shift=0, transform=None, transform_model=None):

        #load the annotations
        self.labels_offence_severity, self.labels_action, self.distribution_offence_severity,self.distribution_action, not_taking = label2vectormerge(path, split, types, num_views)
        #load the path of the clips
        self.clips = clips2vectormerge(path, split, types, num_views, not_taking)

        self.shift = shift
        self.split = split
        self.start = start
        self.end = end
        self.transform = transform
        self.transform_model = transform_model
        self.num_views = num_views

        self.factor = (end - start) / (((end - start) / 25) * fps)

        self.length = len(self.labels_action)

        self.distribution_offence_severity = torch.div(self.distribution_offence_severity, len(self.labels_offence_severity))
        self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

        self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
        self.weights_action = torch.div(1, self.distribution_action)


    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action, 
    def getWeights(self):
        return self.weights_offence_severity, self.weights_action, 

    def __getitem__(self, index):

        frame_shift = random.randint(-self.shift,self.shift)

        prev_views = []

        for num_view in range(len(self.clips[index])):

            index_view = num_view

            if len(prev_views) == 2:
                continue

            cont = True
            if self.split == 'Train':
                while cont:
                    aux = random.randint(0,len(self.clips[index])-1)
                    if aux not in prev_views:
                        cont = False
                index_view = aux
                prev_views.append(index_view)

            video, _, _ = read_video(self.clips[index][index_view], output_format="THWC")

            frames = video[self.start+frame_shift:self.end+frame_shift,:,:,:]

            final_frames = None

            for j in range(len(frames)):
                if j%self.factor<1:
                    if final_frames == None:
                        final_frames = frames[j,:,:,:].unsqueeze(0)
                    else:
                        final_frames = torch.cat((final_frames, frames[j,:,:,:].unsqueeze(0)), 0)

            final_frames = final_frames.permute(0, 3, 1, 2)

            if self.transform != None:
                final_frames = self.transform(final_frames)

            final_frames = self.transform_model(final_frames)
            final_frames = final_frames.permute(1, 0, 2, 3)
            
            if num_view == 0:
                videos = final_frames.unsqueeze(0)
            else:
                final_frames = final_frames.unsqueeze(0)
                videos = torch.cat((videos, final_frames), 0)

        if self.num_views != 1 and self.num_views != 5:
            videos = videos.squeeze()   

        #################
        videos = videos.permute(0, 2, 1, 3, 4)

        # size of mvimages: (Views), Channel, Depth, Height, Width
        return self.labels_offence_severity[index][0], self.labels_action[index][0], videos

    def __len__(self):
        return self.length

