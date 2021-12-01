"""
Designed for SuperGlue_like input
Including: im, kpts, desc, score, ori_shape
GT: Assinment
"""
from datasets.depth_dataset import build_depth    

def build_dataset(args):
    root = '/home/notebook/data/group/zhouyuhao/datasets_personal/DISK-MegaDepth'
    return build_depth(
        root,
        crop_size = (args.height, args.width),
        max_feats = args.max_feats,
        train_limit = args.train_limit,
        test_limit = args.test_limit
    )