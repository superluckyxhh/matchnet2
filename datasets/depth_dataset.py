import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
from os.path import join
import cv2
import numpy as np
import h5py
import json
from scipy.spatial.distance import cdist 

from common.functions import *
from datasets.imageset import ImageSet
from datasets.image import Image
from datasets.limited_dataset import LimitedConcatDataset


"""
MegaDepth   --->  ItemDataset ---> ImageSet     --->   Image
循环所有的scene   得到pose depth    对于每一个scene     写一些属性函数
(135个),经由      后调用Image中     将一对pair的索引    方便调用，例如投影
ItemDataset处理  函数进行分配矩阵   拿出，得到im,pose    反投影，获取点的深度值
拿出需要的值      的制作            depth等

"""

#### Test function ####
def plot_img_kpts(im, kpts, name):
    savp = '/home/notebook/code/personal/S9040697/nxhMatchNet/plot_test'
    # RGB --> Gray
    alpha, beta, gamma = 0.299, 0.587, 0.114
    im = np.array(im) * 255.
    gray = alpha * im[0] + beta * im[1] + gamma * im[2]

    gray = gray.astype(np.uint8)
    H, W = gray.shape
    out = np.ones((H, W, 3), dtype='uint8')
    out[:, :, :] = np.dstack([gray]*3)

    kpts= np.round(np.array(kpts)).astype(int) #[1024, 2]
    for (x, y) in kpts:
        cv2.circle(out, (x, y), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    cv2.imwrite(join(savp, name + '.jpg'), out)
 

#### Test function ####
def plot_depthmap(depth, name):
    savp = '/home/notebook/code/personal/S9040697/nxhMatchNet/plot_test'
    new_depth = torch.where(torch.isnan(depth), torch.full_like(depth, 0), depth)[0]
    new_depth = np.array(new_depth)
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(new_depth, alpha=15), cv2.COLORMAP_JET)
    cv2.imwrite(join(savp, name + '.jpg'), im_color)


#### Test function ####
def plot_project_points(image, pts, proj_pts, name):
    savp = '/home/notebook/code/personal/S9040697/nxhMatchNet/plot_test'
    red = (0, 0, 255)
    green = (0, 255, 0)
    # RGB --> Gray
    alpha, beta, gamma = 0.299, 0.587, 0.114
    im = np.array(image) * 255.
    gray = alpha * im[0] + beta * im[1] + gamma * im[2]

    gray = gray.astype(np.uint8)
    H, W = gray.shape
    out = np.ones((H, W, 3), dtype='uint8')
    out[:, :, :] = np.dstack([gray]*3)

    pts = np.round(np.array(pts)).astype(int)
    proj_pts = np.round(np.array(proj_pts)).astype(int)
    
    # ori pts make red
    for (x, y) in pts:
        cv2.circle(out, (x, y), 1, red, -1, lineType=cv2.LINE_AA)
    # proj pts make green
    for (x1, y1) in proj_pts:
        # ignore nan depth
        if x1 == -9223372036854775808:
            continue
        cv2.circle(out, (x1, y1), 1, green, -1, lineType=cv2.LINE_AA)
    
    cv2.imwrite(join(savp, name + '.jpg'), out)


#### Test function ####
def RGB2Gray(im):
    alpha, beta, gamma = 0.299, 0.587, 0.114
    im = np.array(im) * 255.
    gray = alpha * im[0] + beta * im[1] + gamma * im[2]

    return gray


#### Test function ####
def plot_matches(im0, im1, kpts0, kpts1, matches, name):
    savp = '/home/notebook/code/personal/S9040697/nxhMatchNet/plot_test/signal_matches'
    im0 = RGB2Gray(im0)
    im1 = RGB2Gray(im1)
    kpts0 = np.round(np.array(kpts0)).astype(int)
    kpts1 = np.round(np.array(kpts1)).astype(int)
    h0, w0 = im0.shape[:2]
    h1, w1 = im1.shape[:2]
    margin = 10
    H, W = max(h0, h1), w0 + w1 + margin
    red = (0, 0, 255)
    green = (0, 255, 0)
    
    
    for k in range(len(matches)):
        out = np.ones((H, W, 3), dtype=np.uint8)
        out[:H, :w0] = np.dstack([im0] * 3)
        out[:H, w0 + margin:] = np.dstack([im1] * 3)
        ids0 = matches[:, 0][k]
        ids1 = matches[:, 1][k]
        x0, y0 = kpts0[ids0]
        x1, y1 = kpts1[ids1]
        x1, y1 = x1 + margin + w0, y1

        cv2.circle(out, (x0, y0), 2, red, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1, y1), 2, red, -1, lineType=cv2.LINE_AA)
        cv2.line(out, (x0, y0), (x1, y1), green, 1, lineType=cv2.LINE_AA)
        cv2.imwrite(join(savp, name + str(k) + '.jpg'), out)
        

def _compute_interpolation_size(ori_size, cnt_size):
    x_factor = ori_size[0] / cnt_size[0]
    y_factor = ori_size[1] / cnt_size[1]

    f = 1 / max(x_factor, y_factor)
    if x_factor > y_factor:
        new_size = (cnt_size[0], int(f * ori_size[1]))
    else:
        new_size = (int(f * ori_size[0]), cnt_size[1])
    return f, new_size


def _rescale(points, ori_size, cnt_size):
    _, new_size = _compute_interpolation_size(ori_size, cnt_size)
    points[:, 0] = points[:, 0] * new_size[1] / ori_size[1]
    points[:, 1] = points[:, 1] * new_size[0] / ori_size[0]

    return points


def _crop(image:Image, points:torch.Tensor, crop_size):
    ori_size = image.shape
    # Return image include scaled im, mask, depth, K and origin R T
    image = image.scale(crop_size) 
    cnt_size = image.shape

    points = _rescale(points, ori_size, cnt_size)
    image = image.pad(crop_size)

    return image, points


# Make assignment matrix
def cross_nearest_matching(
    image0:Image, points0:torch.Tensor,
    image1:Image, points1:torch.Tensor,
    th, ambiguous_th=15 * np.sqrt(2)
):  
    # make dustbin row/col like SuperGlue
    N1, N2 = points0.size(0) + 1, points1.size(0) + 1

    ## im0 ---> im1
    # 1. Project points0 from image0 to image1
    points0_proj1 = image1.project(image0.unproject(points0.T)).T
    dists_01 = cdist(points0_proj1.numpy(), points1.numpy(), metric='euclidean')
    dists_01[np.isnan(dists_01)] = float('inf') 

    #2. Select min dists within two keypoints (1v1)
    min1 = np.argmin(dists_01, axis=0)
    min2 = np.argmin(dists_01, axis=1)
    min1v = np.min(dists_01, axis=1)
    min1f = min2[min1v < th]

    xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
    mids1 = np.intersect1d(min1f, xx)
    mids0 = min1[mids1]
    matches = np.concatenate(
        [mids0[np.newaxis, :], mids1[np.newaxis, :]]
    ).T
    n_matches = len(matches)

    #3. Get unmatched points 
    unmatched0 = np.where((min1v > ambiguous_th) & (~np.isinf(min1v)))[0]
    if len(unmatched0) > n_matches // 3:  
        # Make data balanced
        unmatched0 = np.random.choice(unmatched0, size=n_matches//3)

    ## im1 --> im0
    points1_proj0 = image0.project(image1.unproject(points1.T)).T
    dists_10 = cdist(points1_proj0.numpy(), points0.numpy(), metric='euclidean')
    dists_10[np.isnan(dists_10)] = float('inf')
    min0v = np.min(dists_10, axis=1)

    unmatched1 = np.where((min0v > ambiguous_th) & ~np.isinf(min0v))[0]
    unmatched1 = np.setdiff1d(unmatched1, matches[:, 1])

    if len(unmatched1) > n_matches // 3:
        unmatched1 = np.random.choice(unmatched1, size=n_matches//3)

    # assign matches to [N1 x N2] matrix
    assignment = torch.zeros((N1, N2)).bool()
    assignment[unmatched0, -1] = True
    assignment[-1, unmatched1] = True
    if n_matches > 0:
        assignment[matches[:, 0], matches[:, 1]] = True

    ####### Test plot projection ####### 
    # bitmap0 = image0.bitmap
    # bitmap1 = image1.bitmap
    # plot_project_points(bitmap1, points1, points0_proj1, 'projection_im1')
    # plot_project_points(bitmap0, points0, points1_proj0, 'projection_im0')
    ####################################

    return assignment, matches


class ItemDataset:
    def __init__(self, root, json_data, crop_size, max_feats, th=5*np.sqrt(2)):
        self.max_feats = max_feats
        self.crop_size = crop_size
        self.th = th
        # self.ori_im0_shapes = []
        # self.ori_im1_shapes = []
        self.pairs = json_data['pairs']
        self.imageset = ImageSet(root, json_data)
    
    def _get_feats(self, idx):
        name = self.imageset.id2name[idx]
        _name = os.path.splitext(name)[0]
        feat_path = join(self.imageset.feat_path, _name + '.h5' )
        
        with h5py.File(feat_path, 'r') as f:
            descriptors = f['descriptors'][:].astype(np.float32)
            keypoints = f['keypoints'][:].astype(np.float32)
            scores = f['scores'][:].astype(np.float32)
        
        if self.max_feats != -1:
            sort_ids = np.argsort(scores)[::-1]
            keypoints = keypoints[sort_ids][:self.max_feats]
            descriptors = descriptors[sort_ids][:self.max_feats]
            scores = scores[sort_ids][:self.max_feats]
        
        keypoints = torch.from_numpy(keypoints).float()
        descriptors = torch.from_numpy(descriptors).float()
        scores = torch.from_numpy(scores).float()
        
        return keypoints, descriptors, scores


    def __len__(self):
        return len(self.pairs)

    
    def __getitem__(self, idx):
        pair0, pair1 = self.pairs[idx]
        # image0 inclue Image class property (KRT, depth, bitmap, K_inv, shape, hwc, length)
        # image0 could call Image class function like scale ...
        image0 = self.imageset[pair0]
        image1 = self.imageset[pair1]
        ori_im0_shape = image0.orishape
        ori_im1_shape = image1.orishape

        # self.ori_im0_shapes.append(ori_im0_shape)
        # self.ori_im1_shapes.append(ori_im1_shape)

        kpts0, desc0, scores0 = self._get_feats(pair0)
        kpts1, desc1, scores1 = self._get_feats(pair1)

        # Crop image(...) and kpts
        image0, kpts0 = _crop(image0, kpts0, self.crop_size)
        image1, kpts1 = _crop(image1, kpts1, self.crop_size)
        bitmap0 = image0.bitmap
        bitmap1 = image1.bitmap

        ##################  Test: plot crop img, depth ##################
        # depth0 = image0.depth
        # depth1 = image1.depth 
        # plot_img_kpts(bitmap0, kpts0, 'bitamp0_kpts0_f')
        # plot_img_kpts(bitmap1, kpts1, 'bitmap1_kpts1_f')
        # plot_depthmap(depth0, 'depth0')
        # plot_depthmap(depth1, 'depth1')
        # print()
        ################################################################# 

        # Using images and kpts to obtain assignment
        assignment, matches = cross_nearest_matching(image0, kpts0, image1, kpts1, self.th)
        
        if matches.shape[0] < self.max_feats / 50:
            return None
        ##################  Test: plot matches ##################
        # plot_matches(bitmap0, bitmap1, kpts0, kpts1, matches, 'matches')
        #########################################################

        return {
            'bitmap0': bitmap0,
            'bitmap1': bitmap1,
            # 'ori_im0_shapes': torch.tensor(self.ori_im0_shapes),
            # 'ori_im1_shapes': torch.tensor(self.ori_im1_shapes),
            'ori_im0_shapes':ori_im0_shape,
            'ori_im1_shapes':ori_im1_shape,
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'descriptors0': desc0,
            'descriptors1': desc1,
            'scores0': scores0,
            'scores1': scores1,
            'assignment': assignment,
        }


class DepthDataset(LimitedConcatDataset):
    def __init__(
        self, json_path, crop_size=(480, 640), max_feats=1024,
        limit=None, shuffle=False, warn=True
    ):   
        root, _ = os.path.split(json_path)
        
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)

        scene_datasets = []
        for scene in json_data:
            scene_datasets.append(ItemDataset(
                root, 
                json_data[scene], 
                crop_size, 
                max_feats=max_feats
            ))

        super(DepthDataset, self).__init__(
            scene_datasets,
            limit=limit,
            shuffle=shuffle,
            warn=warn
        )

    @staticmethod
    def collate_fn(batch):
        batched_data = list(filter(lambda b: b is not None, batch))
        return torch.utils.data.dataloader.default_collate(batched_data)

    
def build_depth(
    root, 
    crop_size=(480, 640),
    max_feats=1024,
    train_limit=1000,
    test_limit=500
):
    train_dataset = DepthDataset(
        join(root, 'train/dataset.json'),
        crop_size=crop_size,
        max_feats=max_feats,
        limit=train_limit,
        shuffle=True
    )

    # TODO: Test 还没测试
    # TODO: ImageSet pairs ---> tuples
    # test_dataset = DepthDataset(
    #     os.path.join(root, 'test/dataset.json'),
    #     crop_size=crop_size,
    #     max_feats=max_feats,
    #     limit=test_limit,
    #     shuffle=True
    # )
    test_dataset = None

    return train_dataset, test_dataset