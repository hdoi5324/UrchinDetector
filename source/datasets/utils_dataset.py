import os
import torchvision
from .presets import DetectionPresetTrain, DetectionPresetEval



def get_transform(train, opt_train):
    if train:
        return DetectionPresetTrain(data_augmentation=opt_train.data_augmentation)
    elif opt_train.weights and opt_train.test_only:
        weights = torchvision.models.get_weight(opt_train.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return DetectionPresetEval()

