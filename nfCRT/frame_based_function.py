# Copyright (c) 2022, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the Neural Field CRT Dynamic Imaging Library. For more information and source code
# availability see https://github.com/precise-wustl/nf_crt_dynamic_imaging.
#
# Neural Field CRT Dynamic Imaging is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.
import numpy as np
import torch

class FrameBasedFunction:
    def __init__(self, frame_dict, dev, decode_func = None):
        self.times = np.fromiter(frame_dict.keys(), dtype = int)
        self.s = frame_dict[self.times[0]].shape[0]
        
        self.arr = torch.zeros(self.s*len(self.times), device = dev)
        for i, time in enumerate(self.times):
            self.arr[self.s*i:(i+1)*self.s] = torch.from_numpy(frame_dict[time]).float().to(dev)
        
        self.dev = dev
        self.decode_func =  decode_func
    def decode(self, inp):
        if self.decode_func is None:
            int_locs = np.in1d(inp, self.times).nonzero()[0]
            out_locs = np.argwhere(inp[int_locs] == self.times.reshape(-1,1))
            out_locs = out_locs[out_locs[:,1].argsort()][:,0]
            return int_locs, out_locs
        else:
            return self.decode_func(inp)

    def __call__(self, inds):
        N = len(inds)
        evalout = torch.zeros(N, device = self.dev)
        time_coords = inds//self.s
        
        int_locs, out_locs = self.decode(time_coords)

        x_coords = inds%self.s

        evalout[int_locs] = self.arr[self.s*out_locs   + x_coords[int_locs]]
        return evalout