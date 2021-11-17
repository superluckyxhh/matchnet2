import torch
import numpy as np
import os
from os.path import join
import h5py
import imageio
from datasets.image import Image

class ImageSet:
    def __init__(self, root, json_data):
        self.depth_path = join(root, json_data['depth_path'])
        self.image_path = join(root,json_data['image_path'])
        self.calib_path = join(root,json_data['calib_path'])
        self.feat_path = join(root,json_data['feature_path'])
        self.id2name = json_data['images']

        self._check_path(self.depth_path)
        self._check_path(self.image_path)
        self._check_path(self.calib_path)
        self._check_path(self.feat_path)
    
    
    def _check_path(self, path):
        if not os.path.exists(path):
            info = f"Couldn't find the path at {path}"
            raise FileNotFoundError(info)
            
    
    def get_depth(self, name):
        _name = os.path.splitext(name)[0]
        depth_path = join(self.depth_path, _name + '.h5')
        rd = h5py.File(depth_path, 'r')
        depth = rd['depth'][:].astype(np.float32)
        # Process no depth 
        depth[depth == 0.] = float('NaN')
        
        return torch.from_numpy(depth).unsqueeze(0)
        
    
    def get_calib(self, name):
        _name = 'calibration_' + name + '.h5'
        calib_path = join(self.calib_path, _name)
        values = []
        with h5py.File(calib_path, 'r') as calib_file:
            for f in ['K', 'R', 'T']:
                v = torch.from_numpy(calib_file[f][()]).to(torch.float32)
                values.append(v)

        return values
    

    def get_bitmap(self, name):
        im_path = join(self.image_path, name)
        bitmap = imageio.imread(im_path)
        bitmap = bitmap.astype(np.float32) / 255.
        bitmap = torch.from_numpy(bitmap).permute(2, 0, 1)

        return bitmap
    

    def __getitem__(self, idx):
        imn = self.id2name[idx]

        image = Image(
                *self.get_calib(imn),
                self.get_bitmap(imn),
                self.get_depth(imn)
        )

        return image