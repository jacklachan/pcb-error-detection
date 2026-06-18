import random
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


class PCBTransform:
    """
    Paired transform for template and test PCB images.
    Applies identical geometric augmentations to both images to preserve alignment.
    """

    def __init__(self, size=(512, 512), train=True):
        self.size = size
        self.train = train
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, template, test):
        template = TF.resize(template, self.size)
        test = TF.resize(test, self.size)

        if self.train:
            if random.random() > 0.5:
                template = TF.hflip(template)
                test = TF.hflip(test)

            if random.random() > 0.5:
                template = TF.vflip(template)
                test = TF.vflip(test)

            # Same photometric jitter on both
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            template = TF.adjust_brightness(template, brightness)
            template = TF.adjust_contrast(template, contrast)
            test = TF.adjust_brightness(test, brightness)
            test = TF.adjust_contrast(test, contrast)

        template = self.normalize(TF.to_tensor(template))
        test = self.normalize(TF.to_tensor(test))

        return template, test
