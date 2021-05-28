
"""
Adapted from: https://github.com/Britefury/cutmix-semisup-seg/blob/44e81b3ae862d2da7d1c4df77fb274f8f1f0a861/optim_weight_ema.py

Copyright (c) 2020 University of East Anglia, Norwich, UK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

import torch


class EMAWeightOptimizer (object):
    def __init__(self, target_net, source_net, initialize_to_source=True):
        self.target_net = target_net
        self.source_net = source_net
        #self.ema_alpha = ema_alpha

        # for p in target_net.parameters():
        #     p.requires_grad = False

        self.target_params = [p for p in target_net.state_dict().values() if p.dtype == torch.float]
        self.source_params = [p for p in source_net.state_dict().values() if p.dtype == torch.float]

        if initialize_to_source:
            for tgt_p, src_p in zip(self.target_params, self.source_params):
                tgt_p[...] = src_p[...]

        target_keys = set(target_net.state_dict().keys())
        source_keys = set(source_net.state_dict().keys())
        if target_keys != source_keys:
            raise ValueError('Source and target networks do not have the same state dict keys; do they have different architectures?')

    def step(self, ema_alpha):
        one_minus_alpha = 1.0 - ema_alpha
        for tgt_p, src_p in zip(self.target_params, self.source_params):
            tgt_p.mul_(ema_alpha)
            tgt_p.add_(src_p * one_minus_alpha)
