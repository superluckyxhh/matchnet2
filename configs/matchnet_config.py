import types

def get_args_parser():
    args = types.SimpleNamespace()

    #--------------- Paths ---------------#
    args.artifact = '/home/notebook/code/personal/match2weights'
    args.load = None
    #---------- Dataset Params -----------#
    args.batch_size = 4
    args.width = 640
    args.height = 480
    args.max_feats = 2048
    args.train_limit = 200
    args.test_limit = 500
    #----------- Model Params ------------#
    args.feature_dim = 256
    args.kpts_encoder = [32, 64, 128, 256]
    args.num_layers = 9
    args.backbone = 'resnet34'
    args.module_name = 'gnn'
    ###[gemformer1d, gemformer2d, crossAttn, pool, gnn, lineartrans]
    args.loss = 'mean' #['mean', 'log']
    args.score_type = 'DS' #['DS', 'OT']
    #----------- Train Params ------------#
    args.seed = 777
    args.n_epochs = 3
    args.lr = 3e-4
    args.weight_decay = 1e-5
    args.print_freq = 8
    args.save_interval = 1
    args.clip_max_norm = 0.0
    args.eps = 1e-6
    #---------- Distributed Training Params ----------#
    args.world_size = 1
    args.dist_url = 'env://'

    return args