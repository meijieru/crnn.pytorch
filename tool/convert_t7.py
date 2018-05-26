import torchfile
import argparse
import torch
from torch.nn.parameter import Parameter
import numpy as np
import models.crnn as crnn


layer_map = {
    'SpatialConvolution': 'Conv2d',
    'SpatialBatchNormalization': 'BatchNorm2d',
    'ReLU': 'ReLU',
    'SpatialMaxPooling': 'MaxPool2d',
    'SpatialAveragePooling': 'AvgPool2d',
    'SpatialUpSamplingNearest': 'UpsamplingNearest2d',
    'View': None,
    'Linear': 'linear',
    'Dropout': 'Dropout',
    'SoftMax': 'Softmax',
    'Identity': None,
    'SpatialFullConvolution': 'ConvTranspose2d',
    'SpatialReplicationPadding': None,
    'SpatialReflectionPadding': None,
    'Copy': None,
    'Narrow': None,
    'SpatialCrossMapLRN': None,
    'Sequential': None,
    'ConcatTable': None,  # output is list
    'CAddTable': None,  # input is list
    'Concat': None,
    'TorchObject': None,
    'LstmLayer': 'LSTM',
    'BiRnnJoin': 'Linear'
}


def torch_layer_serial(layer, layers):
    name = layer[0]
    if name == 'nn.Sequential' or name == 'nn.ConcatTable':
        tmp_layers = []
        for sub_layer in layer[1]:
            torch_layer_serial(sub_layer, tmp_layers)
        layers.extend(tmp_layers)
    else:
        layers.append(layer)


def py_layer_serial(layer, layers):
    """
    Assume modules are defined as executive sequence.
    """
    if len(layer._modules) >= 1:
        tmp_layers = []
        for sub_layer in layer.children():
            py_layer_serial(sub_layer, tmp_layers)
        layers.extend(tmp_layers)
    else:
        layers.append(layer)


def trans_pos(param, part_indexes, dim=0):
    parts = np.split(param, len(part_indexes), dim)
    new_parts = []
    for i in part_indexes:
        new_parts.append(parts[i])
    return np.concatenate(new_parts, dim)


def load_params(py_layer, t7_layer):
    if type(py_layer).__name__ == 'LSTM':
        # LSTM
        all_weights = []
        num_directions = 2 if py_layer.bidirectional else 1
        for i in range(py_layer.num_layers):
            for j in range(num_directions):
                suffix = '_reverse' if j == 1 else ''
                weights = ['weight_ih_l{}{}', 'bias_ih_l{}{}',
                           'weight_hh_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(i, suffix) for x in weights]
                all_weights += weights

        params = []
        for i in range(len(t7_layer)):
            params.extend(t7_layer[i][1])
        params = [trans_pos(p, [0, 1, 3, 2], dim=0) for p in params]
    else:
        all_weights = []
        name = t7_layer[0].split('.')[-1]
        if name == 'BiRnnJoin':
            weight_0, bias_0, weight_1, bias_1 = t7_layer[1]
            weight = np.concatenate((weight_0, weight_1), axis=1)
            bias = bias_0 + bias_1
            t7_layer[1] = [weight, bias]
            all_weights += ['weight', 'bias']
        elif name == 'SpatialConvolution' or name == 'Linear':
            all_weights += ['weight', 'bias']
        elif name == 'SpatialBatchNormalization':
            all_weights += ['weight', 'bias', 'running_mean', 'running_var']

        params = t7_layer[1]

    params = [torch.from_numpy(item) for item in params]
    assert len(all_weights) == len(params), "params' number not match"
    for py_param_name, t7_param in zip(all_weights, params):
        item = getattr(py_layer, py_param_name)
        if isinstance(item, Parameter):
            item = item.data
        try:
            item.copy_(t7_param)
        except RuntimeError:
            print('Size not match between %s and %s' %
                  (item.size(), t7_param.size()))


def torch_to_pytorch(model, t7_file, output):
    py_layers = []
    for layer in list(model.children()):
        py_layer_serial(layer, py_layers)

    t7_data = torchfile.load(t7_file)
    t7_layers = []
    for layer in t7_data:
        torch_layer_serial(layer, t7_layers)

    j = 0
    for i, py_layer in enumerate(py_layers):
        py_name = type(py_layer).__name__
        t7_layer = t7_layers[j]
        t7_name = t7_layer[0].split('.')[-1]
        if layer_map[t7_name] != py_name:
            raise RuntimeError('%s does not match %s' % (py_name, t7_name))

        if py_name == 'LSTM':
            n_layer = 2 if py_layer.bidirectional else 1
            n_layer *= py_layer.num_layers
            t7_layer = t7_layers[j:j + n_layer]
            j += n_layer
        else:
            j += 1

        load_params(py_layer, t7_layer)

    torch.save(model.state_dict(), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert torch t7 model to pytorch'
    )
    parser.add_argument(
        '--model_file',
        '-m',
        type=str,
        required=True,
        help='torch model file in t7 format'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='output file name prefix, xxx.py xxx.pth'
    )
    args = parser.parse_args()

    py_model = crnn.CRNN(32, 1, 37, 256, 1)
    torch_to_pytorch(py_model, args.model_file, args.output)
