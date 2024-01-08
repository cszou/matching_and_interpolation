#!/usr/bin/env python3
import torch
import sys
import numpy as np
sys.path.append('./src')
from validate_with_imagenet import *
import os
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imagenet_activations_step(directory, step, activ_func, arch="alexnet"):
    state_dict = recreate_state_dict(
                    directory = directory,
                    step = step)
    model = models.__dict__[arch]()
    model.load_state_dict(state_dict)
    model.to(device).eval()
    activations = get_model_activations(model)
    callback = activations_callback_over_imagenet(
                model,
                callback = activ_func,
                use_tqdm = True,
                device = device)
    imagenet_activ = callback(activations)
    for title in imagenet_activ[0].keys():
        output = np.concatenate([activ[title] for activ in imagenet_activ])
        np.save(os.path.join(args.output, f"{step}.{title}.npy"), output)
    #imagenet_activ = np.concatenate(imagenet_activ)
    #np.save(os.path.join(args.output, f"{step}.npy"), imagenet_activ)

def imagenet_activations(directory, activ_func):
    steps = np.unique([int(file.split(".")[0]) for file in os.listdir(directory)])
    for step in steps:
        print("(imagenet_activations) :: step ", step)
        imagenet_activations_step(
            directory,
            step,
            activ_func)


if __name__ == "__main__":

    version_number = '0.1'
    print("Version Number: " + version_number)
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("--directory")
    parser.add_argument("--activ-func", default="center-neuron", choices=["center-neuron", "channel", "multi", "all"])
    parser.add_argument("--activ-func-args", default={}, type=str)
    parser.add_argument("--output")
    args = parser.parse_args()

    args.activ_func_args = eval(args.activ_func_args)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Build up activ_func
    activ_func_dict = {
        "center-neuron": GrabNeuronActivation,
        "channel": GrabChannelActivation,
        "all": GrabAllActivation
    }

    if args.activ_func == "multi":
        activ_funcs = [activ_func_dict[args.activ_func_args["type"]](**kwargs) for kwargs in args.activ_func_args["args"]]
        activ_func = GrabMultiActivation(activ_funcs)
        print(activ_func)
        print(activ_funcs)
    else:
        activ_func = activ_func_dict[args.activ_func](**args.activ_func_args)

    # Compute the ImageNet activations
    imagenet_activ = imagenet_activations(args.directory, activ_func)
    np.save(args.output, imagenet_activ)

    # Send the result to file
    for step, activ in imagenet_activ:
        np.save(os.path.join(args.output, f"{step}.npy"), activ)
