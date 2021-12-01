import numpy as np
from torch.utils.data import DataLoader 
from os.path import join
from datasets.depth_dataset import DepthDataset
from common.functions import *
from common.plot import viz_matches


json_path = '/home/notebook/data/group/zhouyuhao/datasets_personal/DISK-MegaDepth/train/dataset.json'
dataset = DepthDataset(json_path, crop_size=(480, 640), max_feats=1024)


loader = DataLoader(dataset, batch_size=4, shuffle=True,
    collate_fn=dataset.collate_fn)

for batched_data in loader:
    batch_id = 0

    bitmap0, bitmap1 = batched_data['bitmap0'], batched_data['bitmap1'] 
    kpts0, kpts1 = batched_data['keypoints0'], batched_data['keypoints1']
    desc0, desc1 = batched_data['descriptors0'], batched_data['descriptors1']
    score0, score1 = batched_data['scores0'], batched_data['scores1']
    assignment = batched_data['assignment']
    
    # Take a assignment
    assign = assignment[batch_id].cpu().detach().numpy()
    # Check match ids and unmatch ids
    matched0, matched1 = (np.where(assign[:-1, :-1]))
    unmatched0 = np.where(assign[:, -1])[0]
    unmatched1 = np.where(assign[-1, :])[0]
    assert len(np.intersect1d(matched0, unmatched0)) == 0
    assert len(np.intersect1d(matched0, unmatched0)) == 0

    # Assignment to matches & Check matches
    bitmap0 = bitmap0[batch_id].permute(1, 2, 0).cpu().detach().numpy()
    bitmap0 = (bitmap0 * 255.).astype(np.uint8)

    bitmap1 = bitmap1[batch_id].permute(1, 2, 0).cpu().detach().numpy()
    bitmap1 = (bitmap1 * 255.).astype(np.uint8)

    matches = assignments_to_matches(assignment, use_bins=True)
    batch_mask = matches[:, 0] == batch_id

    kpts0 = kpts0[batch_id]
    kpts1 = kpts1[batch_id]

    mkpts0 = kpts0[matches[:, 1][batch_mask]]
    mkpts1 = kpts1[matches[:, 2][batch_mask]]
    print('Input keypoints: ', len(kpts0))
    print('Matched keypoints: ', len(mkpts0))

    viz_matches(
        bitmap0, mkpts0.cpu().numpy(),
        bitmap1, mkpts1.cpu().numpy(),
        'match.png',
        all_kp1=kpts0.cpu().numpy(),
        all_kp2=kpts1.cpu().numpy()
    )
    print('done !')