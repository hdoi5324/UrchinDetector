from .create_model import create_model


def get_faster_rcnn_model(opt):
    # load network
    print("Creating models")
    kwargs = {"trainable_backbone_layers": opt.train.trainable_backbone_layers}
    if opt.train.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in opt.train.model:
        if opt.train.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = opt.train.rpn_score_thresh
    kwargs["min_size"] = opt.train.min_size
    model = create_model(opt, **kwargs)

    if "min_size" in kwargs.keys():
        model.transform.min_size = (kwargs["min_size"],)

    return model
