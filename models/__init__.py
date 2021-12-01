from models.matchnet import MatchingNet

def build_model(args):
    return MatchingNet(
         feature_dim = args.feature_dim,
         kpt_encoder = args.kpts_encoder,
         num_layers = args.num_layers,
         score_type = args.score_type
    )