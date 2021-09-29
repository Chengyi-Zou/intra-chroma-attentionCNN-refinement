import argparse
import os
from importlib.machinery import SourceFileLoader
import sys

import numpy as np

from models.att_multi import m_norelu_pos

def get_mask():
    return np.load("resources/deploy_mask.npy")


def get_layer(model, name):
    for l in model.layers:
        if l.name == name:
            return l


def luma_conv_branch(model, cf):
    # Layer 1
    x1 = get_layer(model, "x1")
    W1, b_luma1 = x1.get_weights()
    W1_shape = W1.shape
    W1 = np.reshape(W1, [9, W1_shape[-2], W1_shape[-1]])
    W1 = np.swapaxes(W1, 0, 2)
    W_luma1 = np.reshape(W1, [W1_shape[-1], 9 * W1_shape[-2]])

    # Layer 2
    x2 = get_layer(model, "x2")
    W2, b_luma2 = x2.get_weights()
    W2_shape = W2.shape
    W2 = np.reshape(W2, [9, W2_shape[-2], W2_shape[-1]])
    W2 = np.swapaxes(W2, 0, 2)
    W_luma2 = np.reshape(W2, [W2_shape[-1], 9 * W2_shape[-2]])

    return {'w_luma1': W_luma1,
            'b_luma1': b_luma1,
            'w_luma2': W_luma2,
            'b_luma2': b_luma2}


def boundary_branch(model, cf):
    # layer 1
    b1 = get_layer(model, "b1")
    W1, b_bound1 = b1.get_weights()
    W_bound1 = np.zeros([W1.shape[2], W1.shape[1]])
    for i in range(W1.shape[-1]):
        W_bound1[i] = W1[..., i][0]

    # layer 2
    b2 = get_layer(model, "b2")
    W2, b_bound2 = b2.get_weights()
    W_bound2 = np.zeros([W2.shape[2], W2.shape[1]])
    for i in range(W2.shape[-1]):
        W_bound2[i] = W2[..., i][0]

    if cf.scheme == 3:
        scale_bit = (2 ** cf.bit_depth) - 1
        W_bound1 = np.floor((W_bound1 / scale_bit) * (2 ** cf.scale_bound1)).astype('int')
        b_bound1 = np.floor(b_bound1 * (2 ** cf.scale_bound1)).astype('int')

        scale_bound2 = cf.scale_bound1 - cf.shift_bound1
        W_bound2 = np.floor((W_bound2 / (2 ** scale_bound2)) * (2 ** cf.scale_bound2)).astype('int')
        b_bound2 = np.floor(b_bound2 * (2 ** cf.scale_bound2)).astype('int')

    return {'w_boundary1': W_bound1,
            'b_boundary1': b_bound1,
            'w_boundary2': W_bound2,
            'b_boundary2': b_bound2}


def attention_module(model, cf):
    # Att b
    att_b = get_layer(model, "att_b")
    W1, b_att_b = att_b.get_weights()
    W_att_b = np.zeros([W1.shape[2], W1.shape[1]])
    for i in range(W1.shape[-1]):
        W_att_b[i] = W1[..., i][0]

    # Att x
    att_x = get_layer(model, "att_x")
    W1, b_att_x = att_x.get_weights()
    W1 = W1[0]
    W_att_x = np.zeros([W1.shape[2], W1.shape[1]])
    for i in range(W1.shape[-1]):
        W_att_x[i] = W1[..., i][0]

    # Att x1
    att_x1 = get_layer(model, "att_x1")
    W1, b_att_x1 = att_x1.get_weights()
    W1 = W1[0]
    W_att_x1 = np.zeros([W1.shape[2], W1.shape[1]])
    for i in range(W1.shape[-1]):
        W_att_x1[i] = W1[..., i][0]

    if cf.scheme == 3:
        scale_attb = cf.scale_bound2 - cf.shift_bound2
        W_att_b = np.floor((W_att_b / (2 ** scale_attb)) * (2 ** cf.scale_attb)).astype('int')
        b_att_b = np.floor(b_att_b * (2 ** cf.scale_attb)).astype('int')

        scale_attx = cf.scale_luma - cf.shift_luma
        W_att_x = np.floor((W_att_x / (2 ** scale_attx)) * (2 ** cf.scale_attx)).astype('int')
        b_att_x = np.floor(b_att_x * (2 ** cf.scale_attx)).astype('int')

        scale_attx1 = cf.scale_luma - cf.shift_luma
        W_att_x1 = np.floor((W_att_x1 / (2 ** scale_attx1)) * (2 ** cf.scale_attx1)).astype('int')
        b_att_x1 = np.floor(b_att_x1 * (2 ** 23)).astype('int')

    return {'w_att_b': W_att_b,
            'b_att_b': b_att_b,
            'w_att_x': W_att_x,
            'b_att_x': b_att_x,
            'w_att_x1': W_att_x1,
            'b_att_x1': b_att_x1}


def prediction_head(model, cf):
    # Layer 1
    t2 = get_layer(model, "t2")
    W1, b_t2 = t2.get_weights()
    W1_shape = W1.shape
    W1 = np.reshape(W1, [9, W1_shape[-2], W1_shape[-1]])
    W1 = np.swapaxes(W1, 0, 2)
    W_t2 = np.reshape(W1, [W1_shape[-1], 9 * W1_shape[-2]])

    # Layer 2
    out = get_layer(model, "out")
    W1, b_out = out.get_weights()
    W1 = W1[0]
    W_out = np.zeros([W1.shape[2], W1.shape[1]])
    for i in range(W1.shape[-1]):
        W_out[i] = W1[..., i][0]

    if cf.scheme == 3:
        scale_bit = (2 ** cf.bit_depth) - 1
        W_out = np.floor((W_out / (2 ** cf.scale_head_in)) * (2 ** cf.scale_head_out) * scale_bit).astype('int')
        b_out = np.floor(b_out * (2 ** cf.scale_head_out) * scale_bit).astype('int')
        W_out[:, -1] = W_out[:, -1] * (2 ** cf.scale_head_in)

    return {'w_t2': W_t2,
            'b_t2': b_t2,
            'w_out': W_out,
            'b_out': b_out}


def deploy_model(model, cf):
    output = luma_conv_branch(model, cf)
    output.update(boundary_branch(model, cf))
    output.update(attention_module(model, cf))
    output.update(prediction_head(model, cf))
    return output


def write_w(file, w):
    with open(file, 'w') as output_file:
        for row in w:
            output_file.write('{' + '\t'.join(['%.18f,' % i for i in row])[:-1] + '},\n')


def write_b(file, b):
    with open(file, 'w') as output_file:
        output_file.write('{' + '\t'.join(['%.18f,' % i for i in b])[:-1] + '}\n')


def write_w_int(file, w):
    with open(file, 'w') as output_file:
        for row in w:
            output_file.write('{' + ' '.join(['%d,' % i for i in row]) + '},\n')


def write_b_int(file, b):
    with open(file, 'w') as output_file:
        output_file.write('{' + ' '.join(['%d,' % i for i in b]) + '}\n')


def write_model(output, cf):
    deploy_path = "%s/scheme%d" % (cf.deploy_path, cf.scheme)
    if not os.path.exists(deploy_path):
        os.mkdir(deploy_path)

    if not cf.scheme == 3:
        for name in output:
            name_path = "%s/%s.txt" % (deploy_path, name)
            if name[0] == 'w':
                write_w(name_path, output[name])
            else:
                write_b(name_path, output[name])
    else:
        for name in output:
            name_path = "%s/%s.txt" % (deploy_path, name)
            write_w_int(name_path, output[name])
            write_b_int(name_path, output[name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='deploy config file')
    args = parser.parse_args()

    cf_deploy = SourceFileLoader('config_deploy', args.config).load_module()
    cf = SourceFileLoader('config_model', cf_deploy.config).load_module()

    if not os.path.exists(cf_deploy.deploy_path):
        os.mkdir(cf_deploy.deploy_path)

    if cf_deploy.scheme == 1:
        if cf.model == 'norelu_pos':
            model = m_norelu_pos.CrossIntraModel(cf).model
    else:
        raise ValueError('Invalid scheme')

    experiment_path = os.path.join(cf.output_path, cf.model, cf.experiment_name)
    output_path = os.path.join(experiment_path, "multi")
    model.load_weights(output_path + '/weights.hdf5')

    output = deploy_model(model, cf_deploy)
    write_model(output, cf_deploy)
