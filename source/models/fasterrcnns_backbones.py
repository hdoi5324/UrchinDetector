from typing import Optional, Any

import torch.nn as nn
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50, vgg16, VGG16_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, _resnet_fpn_extractor, \
    IntermediateLayerGetter, BackboneWithFPN
from torchvision.models.detection.faster_rcnn import _default_anchorgen, FastRCNNConvFCHead, AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.misc import FrozenBatchNorm2d

"""Methods to create a FasterRCNN model with a specified backbone model.  
Based on pytorch vision code.  Can specify num_classes, weights (for FasterRCNN model), 
weights_backbone (pretrained backbone) and which layers to train starting at end."""


def fasterrcnn_resnet50_fpn_v2_backbone(
        *,
        weights: Optional[FasterRCNN_ResNet50_FPN_V2_Weights] = None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        weights_backbone: Optional[ResNet50_Weights] = None,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any,
) -> FasterRCNN:
    """
    Constructs an improved Faster R-CNN model with a ResNet-50-FPN backbone from `Benchmarking Detection
    Transfer Learning with Vision Transformers <https://arxiv.org/abs/2111.11429>`__ paper.

    .. betastatus:: detection module

    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
    :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn` for more
    details.

    Args:
        weights (:class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
            final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are
            trainable. If ``None`` is passed (the default) this value is set to 3.
        **kwargs: parameters passed to the ``torchvision.models.detection.faster_rcnn.FasterRCNN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights
        :members:
    """
    if isinstance(weights, str):
        weights = eval(weights)
    if isinstance(weights_backbone, str):
        weights_backbone = eval(weights_backbone)
    weights = None if weights is None else weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        if num_classes < len(weights.meta["categories"]):
            # Delete end weights
            weights_dict = weights.get_state_dict(progress=progress)
            keys_to_remove = ['roi_heads.box_predictor.cls_score.weight', 'roi_heads.box_predictor.cls_score.bias',
                              'roi_heads.box_predictor.bbox_pred.weight', 'roi_heads.box_predictor.bbox_pred.bias']
            for k in keys_to_remove:
                del weights_dict[k]
        # num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

    backbone = resnet50(weights=weights_backbone, progress=progress)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, norm_layer=nn.BatchNorm2d)
    rpn_anchor_generator = _default_anchorgen()
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights_dict, strict=False)
    return model


def fasterrcnn_resnet50_fpn_backbone(
        *,
        weights: Optional[FasterRCNN_ResNet50_FPN_V2_Weights] = None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        weights_backbone: Optional[ResNet50_Weights] = None,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any,
) -> FasterRCNN:
    return fasterrcnn_resnet50_features(
        with_fpn=True,
        weights=weights,
        progress=progress,
        num_classes=num_classes,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs)


def fasterrcnn_resnet50_backbone(
        *,
        weights: Optional[FasterRCNN_ResNet50_FPN_V2_Weights] = None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        weights_backbone: Optional[ResNet50_Weights] = None,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any,
) -> FasterRCNN:
    return fasterrcnn_resnet50_features(
        with_fpn=False,
        weights=weights,
        progress=progress,
        num_classes=num_classes,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs)


def fasterrcnn_resnet50_features(
        *,
        with_fpn: bool = False,
        weights: Optional[FasterRCNN_ResNet50_FPN_V2_Weights] = None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        weights_backbone: Optional[ResNet50_Weights] = None,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any,
) -> FasterRCNN:
    """
    Faster R-CNN model with a ResNet-50-FPN backbone from the `Faster R-CNN: Towards Real-Time Object
    Detection with Region Proposal Networks <https://arxiv.org/abs/1506.01497>`__
    paper.

    .. betastatus:: detection module

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.    """
    if isinstance(weights, str):
        weights = eval(weights)
    if isinstance(weights_backbone, str):
        weights_backbone = eval(weights_backbone)
    weights = None if weights is None else weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        if num_classes < len(weights.meta["categories"]):
            # Delete end weights
            weights_dict = weights.get_state_dict(progress=progress)
            keys_to_remove = ['roi_heads.box_predictor.cls_score.weight', 'roi_heads.box_predictor.cls_score.bias',
                              'roi_heads.box_predictor.bbox_pred.weight', 'roi_heads.box_predictor.bbox_pred.bias']
            for k in keys_to_remove:
                del weights_dict[k]
        # num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    if with_fpn:
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
        model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)
    else:
        # select layers that won't be frozen
        trainable_layers = trainable_backbone_layers
        if trainable_layers < 0 or trainable_layers > 5:
            raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
        if trainable_layers == 5:
            layers_to_train.append("bn1")
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        returned_layers = [4]
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        backbone.out_channels = 2048
        roi_pooler = MultiScaleRoIAlign(featmap_names=["3"],
                                        output_size=7,
                                        sampling_ratio=2)
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        model = FasterRCNN(backbone, num_classes=num_classes, roi_pooler=roi_pooler,
                           rpn_anchor_generator=anchor_generator,
                           **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)

    return model


def fasterrcnn_vgg16_backbone(
        *,
        weights=None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        weights_backbone: Optional[VGG16_Weights] = None,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any,
):
    return fasterrcnn_vgg16_backbone_features(
        with_fpn=False,
        weights=weights,
        progress=progress,
        num_classes=num_classes,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs)


def fasterrcnn_vgg16_fpn_backbone(
        *,
        weights=None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        weights_backbone: Optional[VGG16_Weights] = None,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any,
):
    return fasterrcnn_vgg16_backbone_features(
        with_fpn=True,
        weights=weights,
        progress=progress,
        num_classes=num_classes,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs)


def fasterrcnn_vgg16_backbone_features(
        *,
        with_fpn=False,
        weights=None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        weights_backbone: Optional[VGG16_Weights] = None,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any,
) -> FasterRCNN:
    """
    """
    ## Load backbone
    if isinstance(weights_backbone, str):
        weights_backbone = eval(weights_backbone)
    if with_fpn:
        # Based on https://discuss.pytorch.org/t/fpn-with-vgg16-backbone-for-fasterrcnn/163166/2
        vgg = VGG16_features(weights_backbone=weights_backbone, progress=progress,
                             trainable_backbone_layers=trainable_backbone_layers)
        in_channels_list = [128, 256, 512, 512]
        return_layers = {'layer_1': '0', 'layer_2': '1', 'layer_3': '2', 'layer_4': '3'}
        out_channels = 256
        backbone = BackboneWithFPN(vgg, return_layers, in_channels_list, out_channels)
        backbone.out_channels = 256
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512), (32, 64, 128, 256, 512), (32, 64, 128, 256, 512), (32, 64, 128, 256, 512)),
            aspect_ratios=((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)))
        roi_pooler = MultiScaleRoIAlign(featmap_names=["2"], output_size=7, sampling_ratio=2)
        model = FasterRCNN(backbone, #roi_pooler=roi_pooler, rpn_anchor_generator=anchor_generator,
                           num_classes=num_classes, **kwargs)
    else:
        vgg = vgg16(progress=progress, weights=weights_backbone)

        # Freeze layers
        if trainable_backbone_layers is None:
            trainable_backbone_layers = len(vgg.features)
        freeze_before = len(vgg.features) if trainable_backbone_layers == 0 else len(
            vgg.features) - trainable_backbone_layers

        for layer in vgg.features[:freeze_before]:
            for p in layer.parameters():
                p.requires_grad = False
        backbone = nn.Sequential(*list(vgg.features)[:-1])

        backbone.out_channels = 512  # Output channels for VGG16

        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                        output_size=7,
                                        sampling_ratio=2)
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        model = FasterRCNN(backbone, num_classes=num_classes, roi_pooler=roi_pooler,
                           rpn_anchor_generator=anchor_generator,
                           **kwargs)

    return model


class VGG16_features(nn.Module):
    """'vgg16':            ('block5_conv3', 'block4_conv3', 'block3_conv3')
    From https://github.com/MrGiovanni/UNetPlusPlus"""
    def __init__(self, weights_backbone=None, progress=True, trainable_backbone_layers=None):
        super().__init__()
        if isinstance(weights_backbone, str):
            weights_backbone = eval(weights_backbone)
        vgg = vgg16(progress=progress, weights=weights_backbone)
        # Freeze layers before splitting up
        if trainable_backbone_layers is None:
            trainable_backbone_layers = len(vgg.features)
        freeze_before = len(vgg.features) if trainable_backbone_layers == 0 else len(
            vgg.features) - trainable_backbone_layers
        for layer in vgg.features[:freeze_before]:
            for p in layer.parameters():
                p.requires_grad = False

        # Split layers
        # Based on https://github.com/Hanqer/deep-hough-transform(Need to add 1 when segmenting array)
        self.layer_1 = nn.Sequential(vgg.features[:9])
        self.layer_2 = nn.Sequential(vgg.features[9:16])
        self.layer_3 = nn.Sequential(vgg.features[16:23])
        self.layer_4 = nn.Sequential(vgg.features[23:-1])

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        return x


def overwrite_eps(model: nn.Module, eps: float) -> None:
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps
