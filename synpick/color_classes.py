
"""
Small utility functions for displaying segmentation
"""

import torch
import hsluv

class Colorizer(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

        self.colors = torch.zeros(num_classes, 3, dtype=torch.uint8)

        sat = 100
        lum = 40

        # White as background
        self.colors[0] = 255

        for i in range(num_classes - 1):
            hue = i * 360.0 / (num_classes - 1)

            rgb = hsluv.hsluv_to_rgb([hue, sat, lum])

            self.colors[i+1] = (torch.tensor(rgb) * 255.0).byte()

    def colorize(self, input):
        assert input.dim() >= 2
        assert input.dtype == torch.long, f"invalid dtype: {input.dtype}"

        flattened = input.view(-1) # N
        colorized = self.colors[flattened] # N x 3
        colorized = colorized.view(-1, input.shape[-2], input.shape[-1], 3)
        colorized = colorized.permute(0, 3, 1, 2)

        return colorized.view(*input.shape[:-2], 3, *input.shape[-2:])
