from losses.loss import Criterion
from losses.mean_loss import MeanCriterion

def build_criterion(args):
    if args.loss == 'mean':
        criterion = MeanCriterion(eps = args.eps)
        metrics = None
    elif args.loss == 'log':
        criterion = Criterion(eps = args.eps)
        metrics = None
    else:
        print('Invalid Loss type')
        exit(1)
    
    return criterion, metrics