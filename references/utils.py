import torch
import numpy as np
from lucent.optvis.param.spatial import rfft2d_freqs
from lucent.optvis.param.color import to_valid_rgb
from collections import OrderedDict
import os
from itertools import chain

def get_model_layers(model):
    """
    Returns a dict with layer names as keys and layers as values
    """
    assert hasattr(model, "_modules")
    layers = {}
    def get_layers(net, prefix=[]):
        for name, layer in net._modules.items():
            if layer is None:
                continue
            if len(layer._modules) != 0:
                get_layers(layer, prefix=prefix+[name])
            else:
                layers["_".join(prefix+[name])] = layer
    get_layers(model)
    return layers

def get_model_activations(model, require_grad = False):
    layers = get_model_layers(model)

    activations = {layer: None for layer in layers.keys()}
    def get_activation_without_grad(layer_name):
        def hook(model, input, output):
            activations[layer_name] = output.detach()
        return hook
    
    def get_activation_with_grad(layer_name):
        def hook(model, input, output):
            activations[layer_name] = output
        return hook

    for name, layer in layers.items():
        if require_grad:
            layer.register_forward_hook(get_activation_with_grad(name))
        else:
            layer.register_forward_hook(get_activation_without_grad(name))

    return activations

def set_model_activations(model, require_grad = False):
    if not hasattr(model, "activations"):
        model.activations = get_model_activations(model, require_grad=require_grad)
    else:
        print("Model already has attribute 'activations'")

def get_objective_regions(model, objective_regions, require_grad=False, obj_func=None):

    # Setting up the objective function
    if obj_func is None:
        obj_func = lambda x: x
    obj_wrapper = lambda x: obj_func(x) if require_grad else obj_func(x).detach()

    layers = get_model_layers(model)
    activations = {(layer,channel): None for layer,channel,_ in objective_regions}

    # NOTE: The use of a "controller" dictionary allows us shut off hook functions when we
    # are not using them, saving on computational time and storage
    controller = {"on": True}

    def get_objective_region(layer, channel, idx):

        def hook(model, input, output):
            if controller["on"]:
                if idx is None:
                    activations[(layer,channel)] = obj_wrapper(output[:, channel])
                elif max(idx) < output.shape[0]:
                    activations[(layer,channel)] = obj_wrapper(output[idx, channel])
                else:
                    activations[(layer,channel)] = None
            else:
                activations[(layer,channel)] = None

        return hook

    for layer, channel, idx in objective_regions:
        layers[layer].register_forward_hook(get_objective_region(layer, channel, idx))

    return activations, controller

# TODO: Remove this if it's completely useless
def get_objective_regions_old(model, objective_regions, maintain = False, require_grad=True, norm = lambda x: torch.norm(x, p=2)):
    
    layers = get_model_layers(model)
    activations = {(layer,channel): None for layer,channel,_ in objective_regions}
    
    # NOTE: The use of a "controller" dictionary allows us shut off hook functions when we
    # are not using them, saving on computational time and storage
    controller = {"on": True}
    
    def get_objective_region(layer, channel, idx, norm=None):

        if norm is None:
            norm = lambda x: x
        
        # All sorts of functions depending on the arguments to get_objective_regions
        def get_obj(model, input, output):
            activations[(layer,channel)] = norm(output[idx,channel]).detach()
        def get_obj_with_grad(model, input, output):
            activations[(layer,channel)] = norm(output[idx,channel])
        def get_obj_maintain(model, input, output):
            activtions[(layer,channel)] = output[:,channel].detach()
        def get_obj_maintain_with_grad(model, input, output):
            activations[(layer,channel)] = output[:,channel]

        if maintain and require_grad:
            my_obj = get_obj_maintain_with_grad
        elif (not maintain) and require_grad:
            my_obj = get_obj_with_grad
        elif maintain and (not require_grad):
            my_obj = get_obj_maintain
        else:
            my_obj = get_obj

        def hook(model, input, output):
            if controller["on"]:
                activations[(layer,channel)] = my_obj(model, input, output)
            else:
                activations[(layer,channel)] = None

        return hook

    for layer, channel, idx in objective_regions:
        layers[layer].register_forward_hook(get_objective_region(layer, channel, idx, norm=norm))

    return activations, controller

def get_model_activations_hook(*args, **kwargs):
    activations = get_model_activations(*args, **kwargs)
    dummy_image = torch.zeros((1, 3, 224, 224))
    dummy_labels = torch.zeros(1000)
    def hook(layer):
        if layer == "input":
            return dummy_image
        elif layer == "labels":
            return dummy_labels
        elif layer in activations.keys():
            return activations[layer]
        else:
            raise Exception(f"No layer {layer} in model")

    return hook

def get_receptive_field(model, layer, channel, x, y, input_size = 224):

    feature_net = model.features[:layer]

    test_img = torch.ones((1, 3, input_size, input_size), dtype=torch.float, requires_grad=True)
    test_img_activation = feature_net(test_img)

    neuron_activation = test_img_activation[0, channel, x, y]
    neuron_activation.backward()

    grad_img = test_img.grad[0].detach().numpy().transpose(1,2,0)

    return (grad_img != 0).astype(np.float32)

def determine_output_channels(layer):
    try:
        return layer.out_channels
    except:
        return None

def lucent_image_custom(w, h=None, sd=None, batch=None, decorrelate=True,fft=True, channels=None, param_f = None):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]

    if param_f is None:
        param_f = fft_image if fft else pixel_image
    params, image_f = param_f(shape)

    # TODO: Figure out what "channels" is supposed to specify
    if channels:
        output = to_valid_rgb(image_f, decorrelate=False)
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output

def pixel_image_custom(shape, dist, decay_power=1, device=None):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor = dist(*shape).to(device).requires_grad_(True)
    return [tensor], lambda: tensor

def fft_image_custom(shape, dist, device = None):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (batch, channels) + freqs.shape + (2,) # for imaginary components

    spectrum_real_imag_t = dist(*init_val_size, freqs).to(device).requires_grad_(True)

    scale = np.minimum(1.0/freqs, max(h, w)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    scaled_spectrum_t = scale * spectrum_real_imag_t
    if TORCH_VERSION >= "1.7.0":
        import torch.fft
        if type(scaled_spectrum_t) is not torch.complex64:
            scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
        image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm="ortho")
    else:
        import torch
        image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
    image = image[:batch, :channels, :h, :w]
    magic = 4.0 # Magic constant from Lucid
    image = image / magic
    return [spectrum_real_imag_t], lambda: image

def uniform_pixel_dist(low = 0., high = 1.):
    return lambda *shape: torch.empty(*shape).uniform_(low, high).type(torch.FloatTensor)

def bright_dot_in_middle(mu = 0.0, sigma = 1.0):

    import scipy.stats

    def inner(*shape):
        normal_dist = scipy.stats.norm(mu, sigma)
        output = torch.ones(*shape)
        nrows, ncols = output.shape[2:]
        midrow, midcol = nrows // 2, ncols // 2
        for row in range(nrows):
            for col in range(ncols):
                distance = np.linalg.norm([row - midrow, col - midcol])
                output[..., row, col] = normal_dist.pdf(distance)
        return output

    return inner

def frequency_distribution(freq_sigmas):
    pass

def get_layer_parameters(model, layer_name):
    for name, param in model.named_parameters():
        if layer_name.replace("_", ".") + "." in name:
            yield param

def recreate_state_dict(directory, step):
    state_dict = OrderedDict()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        attributes = filename.split(".")
        if int(attributes[0]) != step:
            continue
        key = ".".join(attributes[1:4])
        parameter = torch.load(filepath)
        state_dict[key] = parameter
    return state_dict

def _flatten(arr):
    if type(arr) is list and type(arr[0]) is list:
        return list(chain(*[_flatten(el) for el in arr]))
    else:
        return arr

