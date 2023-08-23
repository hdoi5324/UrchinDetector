from .fasterrcnns_backbones import *

def create_model(opt, **kwargs):
    model_fn = opt.train.model
    kwargs.update({
        'num_classes': opt.train.num_classes,
        'weights': opt.train.weights,
        'weights_backbone': opt.train.weights_backbone,
    })
    if opt.train.rpn_score_thresh is not None:
        kwargs["rpn_score_thresh"] = opt.train.rpn_score_thresh
    if opt.train.min_size is not None:
        kwargs["min_size"] = opt.train.min_size
    model = eval(model_fn)(**kwargs)
    return model
