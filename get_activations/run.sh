#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python get_activations_top100.py
CUDA_VISIBLE_DEVICES=1 python get_activations_top1000.py