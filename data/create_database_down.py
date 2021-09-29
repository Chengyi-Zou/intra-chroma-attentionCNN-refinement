import argparse
import glob
import os

import cv2
import h5py
import numpy as np
import tqdm

import math
from functools import partial

import matplotlib.pyplot as plt

def readyuv420(filename, bitdepth, W, H, startframe,
               totalframe, show=False):
    # 从第startframe（含）开始读（0-based），共读totalframe帧

    uv_H = H // 2
    uv_W = W // 2

    if bitdepth == 8:
        Y = np.zeros((totalframe, H, W), np.uint8)
        U = np.zeros((totalframe, uv_H, uv_W), np.uint8)
        V = np.zeros((totalframe, uv_H, uv_W), np.uint8)
    elif bitdepth == 10:
        Y = np.zeros((totalframe, H, W), np.uint16)
        U = np.zeros((totalframe, uv_H, uv_W), np.uint16)
        V = np.zeros((totalframe, uv_H, uv_W), np.uint16)

    plt.ion()

    bytes2num = partial(int.from_bytes, byteorder='little', signed=False)

    bytesPerPixel = math.ceil(bitdepth / 8)
    seekPixels = startframe * H * W * 3 // 2
    fp = open(filename, 'rb')
    fp.seek(bytesPerPixel * seekPixels)

    for i in range(totalframe):

        for m in range(H):
            for n in range(W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    Y[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    Y[i, m, n] = np.uint16(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    U[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    U[i, m, n] = np.uint16(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    V[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    V[i, m, n] = np.uint16(pel)

        if show:
            print(i)
            plt.subplot(131)
            plt.imshow(Y[i, :, :], cmap='gray')
            plt.subplot(132)
            plt.imshow(U[i, :, :], cmap='gray')
            plt.subplot(133)
            plt.imshow(V[i, :, :], cmap='gray')
            plt.show()
            plt.pause(100)
            # plt.pause(0.001)

    if totalframe == 1:
        return Y[0], U[0], V[0]
    else:
        return Y, U, V

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('-a', '--area_portion', type=float, default=0.5)
    parser.add_argument('-b', '--max_blocks', type=int, default=2000)
    parser.add_argument('-e', '--ext_bound', type=bool, default=False)
    parser.add_argument('-d', '--bit_depth', type=int, default=10)
    parser.add_argument('-nm', '--normalize', type=bool, default=True)
    args = parser.parse_args()

    qps = [22, 27, 32, 37]
    classes = [0, 0, 1, 2, 3, 4, 5, 6]
    block_shapes = [(4, 4), (8, 8), (16, 16)]
    keys = ['org_chroma', 'rec_boundary_y', 'rec_boundary_uv', 'rec_luma']
    path_org = args.input_path + '/DIV2K_yuv'
    path_rec = args.input_path + '/rec'

    output_path = args.output_path + '/%.2f-%d-%d' % (args.area_portion, args.max_blocks, int(args.ext_bound))
    if not os.path.exists(args.output_path): os.mkdir(args.output_path)
    if not os.path.exists(output_path): os.mkdir(output_path)

    with open(output_path + '/config.txt', 'w') as logs:
        logs.write('area portion: %.2f\n' % args.area_portion)
        logs.write('max_blocks: %d\n' % args.max_blocks)
        logs.write('ext_bound: %d\n' % int(args.ext_bound))
        logs.write('bit depth: %d\n' % args.bit_depth)
        logs.write('normalize: %d\n' % int(args.normalize))

        for mode in ['train', 'val']:
            print(mode)
            input_path = os.path.join(path_org, mode)
            input_path_rec = os.path.join(path_rec, mode)
            mode_path = os.path.join(output_path, mode)
            if not os.path.exists(mode_path): os.mkdir(mode_path)

            output = {}
            for shape in block_shapes:
                boundary_size_y = 2 * sum(shape) + 1 if not args.ext_bound \
                    else ((4 * shape[0]) + (4 * shape[1]) + 1)
                boundary_size_uv = 2 * (sum(shape) + 1) if not args.ext_bound \
                    else 2 * ((2 * shape[0]) + (2 * shape[1]) + 1)
                hf = h5py.File('%s/%dx%d.h5' % (mode_path, shape[0], shape[1]), 'w')
                hf.create_dataset('rec_luma', (1, shape[0] * 2, shape[1] * 2, 1), maxshape=(None, shape[0] * 2, shape[1] * 2, 1))
                hf.create_dataset('rec_boundary_y', (1, boundary_size_y), maxshape=(None, boundary_size_y))
                hf.create_dataset('rec_boundary_uv', (1, boundary_size_uv), maxshape=(None, boundary_size_uv))
                hf.create_dataset('org_chroma', (1, shape[0], shape[1], 2), maxshape=(None, shape[0], shape[1], 2))
                output["%dx%d" % shape] = hf

            blocks = {"%dx%d" % b: 0 for b in block_shapes}
            files = glob.glob(os.path.join(input_path, '0/*'))
            files.sort()
            start = int(files[0].split('/')[-1].split('_')[0])
            for file_id in tqdm.tqdm(range(start, start + len(files))):
                cat = np.random.choice(classes, 1)
                file_path = glob.glob(os.path.join(input_path, '{:d}' + '/{:04d}' + '*.yuv').format(cat[0], file_id))
                file_name = file_path[0].split('/')[-1]
                yuv = "{}/{:d}/{}".format(input_path, cat[0], file_name)
                name = file_name.split('.')[0]
                size = name.split('_')[1]
                width = int(size.split('x')[0])//2
                height = int(size.split('x')[1])//2
                y, u, v = readyuv420(yuv, 10, width * 2, height * 2, 0, 1, False)
                recyuv = os.path.join(input_path_rec, '{:d}', name).format(cat[0])
                #for qp in qps:
                qp = np.random.choice(qps, 1)
                recyuv_qp = os.path.join(recyuv, '{}.yuv').format(qp[0])
                recy, recu, recv = readyuv420(recyuv_qp, 10, width * 2, height * 2, 0, 1, False)
                y = y.astype(np.float32)
                u = u.astype(np.float32)
                v = v.astype(np.float32)
                recy = recy.astype(np.float32)
                recu = recu.astype(np.float32)
                recv = recv.astype(np.float32)
                if args.normalize:
                    y /= (2 ** args.bit_depth - 1)
                    u /= (2 ** args.bit_depth - 1)
                    v /= (2 ** args.bit_depth - 1)
                    recy /= (2 ** args.bit_depth - 1)
                    recu /= (2 ** args.bit_depth - 1)
                    recv /= (2 ** args.bit_depth - 1)

                for shape in block_shapes:
                    portion = min(int((height * width * args.area_portion) /
                                      (shape[0] * shape[1])), args.max_blocks)
                    blocks["%dx%d" % shape] += portion
                    lm = 1 if not args.ext_bound else 2
                    xx, yy = np.mgrid[1:height - lm * shape[0]:shape[0],
                             1:width - lm * shape[1]:shape[1]]
                    positions = np.vstack([xx.ravel(), yy.ravel()]).T
                    positions = positions[np.random.choice(np.arange(len(positions)), portion)]
                    data = output["%dx%d" % shape]
                    info = {'rec_boundary_y': [], 'rec_boundary_uv': [], 'rec_luma': [], 'org_chroma': []}
                    for p in positions:
                        if not args.ext_bound:
                            rows_y = np.append(np.array([p[0] * 2 - 1] * (shape[1] * 2 + 1)),
                                             np.arange(p[0] * 2, p[0] * 2 + shape[0] * 2))
                            cols_y = np.append(np.arange(p[1] * 2, p[1] * 2 + shape[1] * 2),
                                             np.array([p[1] * 2 - 1] * (shape[0] * 2 + 1)))
                            rows_uv = np.append(np.array([p[0] - 1] * (shape[1] + 1)),
                                             np.arange(p[0], p[0] + shape[0]))
                            cols_uv = np.append(np.arange(p[1], p[1] + shape[1]),
                                             np.array([p[1] - 1] * (shape[0] + 1)))
                        else:
                            rows = np.append(np.array([p[0] - 1] * (1 + shape[1] * 2)),
                                             np.arange(p[0], p[0] + shape[0] * 2))
                            cols = np.append(np.arange(p[1], p[1] + shape[1] * 2),
                                             np.array([p[1] - 1] * (1 + shape[0] * 2)))
                        # rec boundaries_y
                        values = np.append([], [ch for ch in recy[rows_y, cols_y].T])
                        info['rec_boundary_y'].append(values)
                        # rec boundaries_uv
                        recuv = np.concatenate((np.expand_dims(recu, -1), np.expand_dims(recv, -1)), axis=2)
                        values = np.append([], [ch for ch in recuv[rows_uv, cols_uv, :].T])
                        info['rec_boundary_uv'].append(values)
                        # rec luma
                        values = np.expand_dims(recy[p[0] * 2:p[0] * 2 + shape[0] * 2, p[1] * 2:p[1] * 2 + shape[1] * 2], -1)
                        info['rec_luma'].append(values)
                        # org chroma
                        values_u = np.expand_dims(u[p[0]:p[0] + shape[0], p[1]:p[1] + shape[1]], -1)
                        values_v = np.expand_dims(v[p[0]:p[0] + shape[0], p[1]:p[1] + shape[1]], -1)
                        values = np.concatenate((values_u, values_v), axis=2)
                        info['org_chroma'].append(values)

                    for key in keys:
                        data[key].resize((data[key].shape[0] + len(positions) - 1), axis=0)
                        data[key][-len(positions):] = info[key]
                        if file_id - start < len(files) - 1:
                            data[key].resize((data[key].shape[0] + 1), axis=0)

            logs.write('---\n%s\n' % mode)
            for shape in block_shapes:
                b = "%dx%d" % shape
                logs.write("%s: %d\n" % (b, blocks[b]))
                output[b].close()
