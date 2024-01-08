#!/usr/bin/env python3

from tqdm import tqdm

import torch
from torch import nn
from torchvision import datasets, transforms

import sys
sys.path.append('./src')

_imagenet_dir = "/data/imagenet_data"
_nclasses = 1000
_val_set_len = 50000
_val_dataset = None
_val_loader = None
_train_dataset = None
_train_loader = None
_train_iter = None
_default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_normalize = transforms.Normalize(
                            mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225])
_standard_train_transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                _normalize])
_standard_val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

def calculate_accuracy(outputs, labels):
    return (outputs == labels).sum() / labels.shape[0]

def calculate_confusion(outputs, labels):
    confusion = torch.zeros((_nclasses, _nclasses))
    for output, label in zip(outputs, labels):
        confusion[output.long(), label.long()] += 1
    return confusion

def grab_activations(activations):
    output = {}
    for name, activ in activations.items():
        output[name] = activ.detach().cpu().numpy()
    return output

class Function(object):
    def __init__(self):
        pass
    def __call__(self, x):
        pass

class GrabNeuronActivation(object):
    
    def __init__(self, layer, channel, x=None, y=None):
        self.layer = layer
        self.channel = channel
        self.x = x
        self.y = y

    def inner(self, activations):
        activ = activations[self.layer][:,self.channel]
        nrows, ncols = activ.shape[1:]
        x = self.x or nrows // 2
        y = self.y or ncols // 2
        return activ[:,x,y].detach().cpu().numpy()

    def __call__(self, activations):
        return {self.title(): self.inner(activations)}

    def title(self):
        if self.x is not None and self.y is not None:
            return f"neuron_{self.layer}_{self.channel}_{self.x}_{self.y}"
        else:
            return f"neuron_{self.layer}_{self.channel}"

class GrabChannelActivation(object):

    def __init__(self, layer, channel):
        self.layer = layer
        self.channel = channel

    def inner(self, activations):
        activ = activations[self.layer][:,self.channel]
        return torch.norm(activ, p=2).detach().cpu().numpy()

    def __call__(self, activations):
        return {self.title(): self.inner(activations)}

    def title(self):
        return f"channel_{self.layer}_{self.channel}"

class GrabMultiActivation(object):

    def __init__(self, activ_funcs):
        self.activ_funcs = activ_funcs

    def __call__(self, activations):
        return {activ_func.title(): activ_func.inner(activations) for activ_func in self.activ_funcs}

grab_mapping = {
        "center-neuron": GrabNeuronActivation,
        "neuron": GrabNeuronActivation,
        "channel": GrabChannelActivation,
        }



class GrabAllLayerActivation(GrabMultiActivation):

    def __init__(self, activ_func_type, layer):
        self.activ_func_type = activ_func_type
        self.activ_funcs = None
        self.layer = layer

    def __call__(self, activations):
        if self.activ_funcs is None:
            self.activ_funcs = [grab_mapping[self.activ_func_type](self.layer, i) for i in range(activations[self.layer].shape[1])]
        super(GrabAllLayerActivation, self).__call__(activations)

class GrabAllActivation(GrabMultiActivation):

    def __init__(self, activ_func_type):
        self.activ_func_type = activ_func_type
        self.activ_funcs = None

    def _call__(self, activations):
        if self.activ_funcs is None:
            self.activ_funcs = _flatten([[__dict__[self.activ_func_type](layer, i) for i in range(data.shape[1])] for layer,data in activations.items()])
        super(GrabAllActivation, self).__call__(activations)

def _calculate_accuracy_by_class(outputs, labels):
    pass

def _make_imagenet_val_dataloader(batch_size = 100, num_workers = 10, **kwargs):
    global _val_dataset, _val_loader
    _val_dataset = datasets.ImageNet(_imagenet_dir, split = "val", transform = _standard_val_transform)
    _val_loader = torch.utils.data.DataLoader(_val_dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = True, shuffle = False, **kwargs)

def _make_imagenet_train_dataloader(batch_size = 100, num_workers = 10, **kwargs):
    global _train_dataset, _train_loader
    _train_dataset = datasets.ImageNet(_imagenet_dir, split = "train", transform = _standard_train_transform)
    _train_loader = torch.utils.data.DataLoader(_train_dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = True, shuffle = True, **kwargs)

def _make_train_iter():
    global _train_iter
    assert _train_dataset is not None
    _train_iter = iter(_train_loader)

def validate_with_imagenet(model, criterion = calculate_accuracy, device = _default_device, use_tqdm = False, do_print = False, **kwargs):
    """
    Runs the model against the ImageNet validation set, calculates a criterion, and returns it
    
    @param model: A torch.Module set up to run over ImageNet
    @param criterion: The criterion function (takes outputs and labels as args)
    @param device: Device to run code on
    @param use_tqdm: Whether to use tqdm on the validation loop
    @param do_print: Whether to print the output
    @param **kwargs: keyword arguments to _make_imagenet_val_dataloader
    @return: the criterion run over the outputs the model over the validation set and the labels of the validation set
    """

    optional_tqdm = lambda iterable: tqdm(iterable) if use_tqdm else iterable
    
    if _val_loader is None:
        _make_imagenet_val_dataloader(**kwargs)

    def inner(activations):
        with torch.no_grad():
            all_outputs = torch.zeros(_val_set_len, device=device)
            all_labels = torch.zeros(_val_set_len, device=device)
            for batch_num, (imgs, labels) in enumerate(optional_tqdm(_val_loader)):

                imgs = imgs.to(device)
                labels = labels.to(device)

                batch_size = labels.shape[0]
                outputs = model(imgs)
                outputs = torch.argmax(outputs, dim=1)
                all_outputs[batch_num * batch_size : (batch_num + 1) * batch_size] = outputs
                all_labels[batch_num * batch_size : (batch_num + 1) * batch_size] = labels
            loss = criterion(all_outputs, all_labels)
        if do_print:
            print(loss)
        return loss

    return inner

def activations_callback_over_imagenet(model, callback=grab_activations, device=_default_device, use_tqdm = False, do_print=False, **kwargs):
    """
    Runs the model against ImageNet validation set, calculates a callback based on the activations, and returns it
    
    @param model: A torch.Module set up to run over ImageNet
    @param callback: The callback function (takes activations as input)
    @param device: Device to run code on
    @param use_tqdm: Whether to use tqdm on the validation loop
    @param do_print: Whether to print the output
    @param **kwargs: keyword arguments to _make_imagenet_val_dataloader
    @return: the callback run over the activations the model over the validation set and the labels of the validation set
    """
    
    optional_tqdm = lambda iterable: tqdm(iterable) if use_tqdm else iterable
    
    if _val_loader is None:
        _make_imagenet_val_dataloader(**kwargs)

    def inner(activations):
        with torch.no_grad():
            outputs = []
            for batch_num, (imgs, labels) in enumerate(optional_tqdm(_val_loader)):

                imgs = imgs.to(device)
                labels = labels.to(device)

                model(imgs)

                outputs.append( callback(activations) )
        if do_print:
            print(outputs)
        return outputs

    return inner
    

def maintain_obj_imagenet(model, criterion=nn.CrossEntropyLoss(), device=_default_device, use_tqdm=False, do_print=False, **kwargs):

    criterion = criterion.to(device)

    optional_tqdm = lambda iterable: tqdm(iterable) if use_tqdm else iterable
    
    if _train_loader is None:
        _make_imagenet_train_dataloader(**kwargs)
        _make_train_iter()

    def inner(activations):
        imgs, labels = _train_iter._next_data()
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        return criterion(outputs, labels)

    return inner 

def maintain_softmax_imagenet(model, orig_model, criterion=nn.CrossEntropyLoss(), device=_default_device, use_tqdm=False, **kwargs):
    
    criterion = criterion.to(device)

    optional_tqdm = lambda iterable: tqdm(iterable) if use_tqdm else iterable
    
    softmax_func = nn.Softmax()
    
    if _train_loader is None:
        _make_imagenet_train_dataloader(**kwargs)
        _make_train_iter()

    def inner(activations):
        imgs, _ = _train_iter._next_data()
        imgs = imgs.to(device)
        softmax = model(imgs)
        orig_softmax = softmax_func( orig_model(imgs) )
        return criterion(softmax, orig_softmax)

    return inner 

_orig_model = None
_orig_model_activations = None
def _set_orig_model(orig_model, require_grad=False, activations=None):
    global _orig_model, _orig_model_activations
    _orig_model = orig_model
    if activations is None:
        _orig_model_activations = get_model_activations(orig_model, require_grad=require_grad)
    else:
        _orig_model_activations = activations

def maintain_activation_imagenet_alternative(model, orig_model, criterion, device=_default_device, use_tqdm=False, **kwargs):

    optional_tqdm = lambda iterable: tqdm(iterable) if use_tqdm else iterable

    if _train_loader is None:
        _make_imagenet_train_dataloader(**kwargs)
        _make_train_iter()

    set_model_activations(model, require_grad=True)
    set_model_activations(orig_model)

    def inner(rate = 1.0):
        imgs, _ = _train_iter._next_data()
        imgs = imgs.to(device)
        model(imgs)
        orig_model(imgs)
        loss = criterion(model.activations, orig_model.activations)
        (rate * loss).backward()
        return loss.detach().cpu().numpy()

    return inner


def maintain_activation_imagenet_old(model, orig_model, activ_func=None, criterion=nn.MSELoss(), activ_criterion=None, device=_default_device, use_tqdm=False, activations=None, **kwargs):
    print("USING WRONG MAINTAIN ACTIVATION IMAGENET OLD")
    try:
        criterion = criterion.to(device)
    except AttributeError:
        print("Criterion is not a Module")

    optional_tqdm = lambda iterable: tqdm(iterable) if use_tqdm else iterable
    
    assert (activ_func is not None and criterion is not None) or activ_criterion is not None, "Must have either an activ_func, criterion combination or activ_criterion specified"
    #assert activ_func is not None, "Must specify an activ_func"
    #assert criterion is not None, "Must specify a criterion"
    
    if _train_loader is None:
        _make_imagenet_train_dataloader(**kwargs)
        _make_train_iter()

    if _orig_model_activations is None:
        print("Setting orig model")
        _set_orig_model(orig_model, require_grad=False, activations=activations)
        print(_orig_model_activations)
    
    def inner(activations):
        imgs, _ = _train_iter._next_data()
        imgs = imgs.to(device)
        model(imgs)
        _orig_model(imgs)
        activ = activ_func(activations)
        orig_activ = activ_func(_orig_model_activations)
        return sum(criterion(activ_val, orig_activ_val) for activ_val, orig_activ_val in zip(activ, orig_activ))
        #return sum(sum(criterion(activ_layer[:,i], orig_activ_layer[:,i]) for i in range(activ_layer.shape[1])) for activ_layer, orig_activ_layer in zip(activ_val, orig_activ_val))
        #return criterion(activ_val, orig_activ_val)
        # TODO: Keep this sum(torch.sum(...)) thing in do_optimization.py
        # TODO: This does weird things for maintain-activations-layer, should have activ-func and criterion by one function, or at least have that as an option

    def inner_activ_criterion(activations):
        imgs, _ = _train_iter._next_data()
        imgs = imgs.to(device)
        model(imgs)
        _orig_model(imgs)
        return activ_criterion(activations, _orig_model_activations)

    return inner if activ_criterion is None else inner_activ_criterion


if __name__ == "__main__":

    import sys
    from line_profiler import LineProfiler
    from torchvision import models
    sys.path.append('./src')
    from utils import *

    # Processing command line arguments in a flexible way
    print("Processing cmd")
    kwargs = {}
    for arg in sys.argv[1:]:
        key, value = arg.split("=")
        kwargs[key] = int(value)
    
    print("Loading model")
    vgg19 = models.model(pretrained=True)
    vgg19.to(_default_device).eval()

    print("Modifying weights")
    weight = vgg19.features[10].weight.detach().cpu().numpy()
    bias = vgg19.features[10].bias.detach().cpu().numpy()
    weight[0,...] = 0.0
    bias[0] = 0.0
    vgg19.features[10].weight = torch.nn.Parameter(torch.from_numpy(weight).to(_default_device))
    vgg19.features[10].bias = torch.nn.Parameter(torch.from_numpy(bias).to(_default_device))

    print("Hooking activations")
    activations = get_model_activations(vgg19)

    print("Building validate_with_imagenet")
    validate = validate_with_imagenet(vgg19, use_tqdm = True, **kwargs)
    lp = LineProfiler()
    validate = lp(validate)
    
    print("Running validate_with_imagenet")
    try:
        accuracy = validate(activations)
        print("Accuracy:", accuracy)
    except KeyboardInterrupt:
        print("Output ended")
    lp.print_stats()

    print ("Building activations_callback_over_imagenet")
    neuron_activ = GrabNeuronActivation(layer = "features_10", channel = 0)
    callback = activations_callback_over_imagenet(vgg19, callback = neuron_activ, use_tqdm = True, **kwargs)
    lp = LineProfiler()
    callback = lp(callback)

    print("Running activations_callback_over_imagenet")
    try:
        output = callback(activations)
        print("Output:", output)
    except KeyboardInterrupt:
        print("Output ended")
    lp.print_stats()

    print("Building maintain_obj_imagenet")
    maintain_obj = maintain_obj_imagenet(vgg19, use_tqdm = True, **kwargs)
    lp = LineProfiler()
    maintain_obj = lp(maintain_obj)

    print("Running maintain_obj_imagenet")
    try:
        objective_value = maintain_obj(activations)
        print("Objective value:", objective_value)
    except KeyboardInterrupt:
        print("Output ended")
    lp.print_stats()
    
    print("Building maintain_softmax_imagenet")
    maintain_obj = maintain_softmax_imagenet(vgg19, use_tqdm = True, **kwargs)
    lp = LineProfiler()
    maintain_obj = lp(maintain_obj)
    
    print("Running maintain_softmax_imagenet")
    try:
        objective_value = maintain_obj(activations)
        print("Objective value:", objective_value)
    except KeyboardInterrupt:
        print("Output ended")
    lp.print_stats()
    
    
