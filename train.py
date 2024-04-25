# Code for training the robust monocular depth estimation model, enhanced from the original Monodepth2 codebase.
# Author: Jiyuan Wang
# Created: 2022-10-1
# Origin used for paper: https://arxiv.org/abs/2310.05556v2
# Hope you can cite our paper if you use the code for your research.
from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
import os
import torch

options = MonodepthOptions()
opts = options.parse()

if opts.debug:
    print("set save mode:", opts.save_strategy,"please \033[91mSTOP\033[0m if stll need old model")
if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
