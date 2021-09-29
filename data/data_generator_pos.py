import os
import threading

import numpy as np
from keras.utils.io_utils import HDF5Matrix

keys = ['org_chroma', 'rec_boundary', 'rec_luma']
nodes = {'in': ['rec_luma', 'rec_boundary'], 'out': ['org_chroma']}

block_shapes = ['4x4', '8x8', '16x16']
positions = {}
positions_bou = {}
for block_shape in block_shapes:
    shape = tuple([int(i) for i in block_shape.split('x')])
    position = np.zeros([shape[0], shape[1], 2])
    position_bou = np.zeros([shape[0] + shape[1] + 1, 2])
    for i in range(shape[0]):
        position[i, :, 0] = (i + 1) / shape[0]
    for i in range(shape[1]):
        position[:, i, 1] = (i + 1) / shape[1]
    positions[block_shape] = position
    for i in range(shape[0] + 1):
        position_bou[i, 0] = (shape[0] - i) / shape[0]
        position_bou[i, 1] = 0
    for i in range(shape[1]):
        position_bou[shape[0] + 1 + i, 0] = 0
        position_bou[shape[0] + 1 + i, 1] = (i + 1) / shape[1]
    positions_bou[block_shape] = position_bou

class DIV2KDataset:
    def __init__(self, data_path,
                 mode='train',
                 block_shape='4x4',
                 batch_size=32,
                 chunk_size=30000,
                 hint=0,
                 samples=0,
                 shuffle=False,
                 get_vol=False,
                 bypass=False,
                 seed=42):

        assert mode in ['train', 'val']
        data_path = os.path.join(data_path, mode, "%s.h5" % block_shape)

        assert os.path.exists(data_path)

        self._data_in = [HDF5Matrix(data_path, k) for k in nodes['in']]
        self._data_out = [HDF5Matrix(data_path, k) for k in nodes['out']]
        self.samples = self._data_in[0].shape[0]
        if samples < self.samples and samples != 0: self.samples = samples

        self.block_shape_dict = block_shape
        self.block_shape = tuple([int(i) for i in block_shape.split('x')])
        self.bound_size = int(self._data_in[-1].shape[-1] / 3)
        self.bound_mid = int(self.bound_size // 2)

        self.get_vol = get_vol
        self.bypass = bypass

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        chunk_size = np.minimum(chunk_size, self.samples)
        self.nb_steps = self.samples // self.batch_size
        self._chunk_size = chunk_size // self.batch_size * self.batch_size
        self._nb_chunks = self.samples // self._chunk_size

        self.hint = hint

        self._lock = threading.Lock()
        self._total_batches_seen = 0
        self._step = 0
        self._index_generator = self._flow_index()

    def bound2vol(self, batch):
        out = np.zeros([batch.shape[0], self.bound_size, 3])
        for ch in range(3):
            b_ch = batch[:, ch * self.bound_size:(ch + 1) * self.bound_size]
            out[:, 0:self.bound_mid + 1, ch] = np.flip(b_ch[:, self.bound_mid:], 1)
            out[:, self.bound_mid + 1:, ch] = b_ch[:, 0:self.bound_mid]
        return out

    def add_position(self, batch):
        out = np.zeros([batch.shape[0], self.block_shape[0], self.block_shape[1], 3])
        out[:, :, :, 0] = batch[:, :, :, 0]
        out[:, :, :, 1] = positions[self.block_shape_dict][:, :, 0]
        out[:, :, :, 2] = positions[self.block_shape_dict][:, :, 1]
        return out

    def add_position_bou(self, batch):
        out = np.zeros([batch.shape[0], self.block_shape[0] + self.block_shape[1] + 1, 5])
        out[:, :, 0:3] = batch
        out[:, :, 3:5] = positions_bou[self.block_shape_dict]
        return out

    def restart(self):
        self._total_batches_seen = 0
        self._step = 0
        self._index_generator = self._flow_index()

    def _flow_index(self):
        while True:
            for self.chunk_index in range(self._nb_chunks):
                np.random.seed(self.seed + self._total_batches_seen)
                index_array = np.random.permutation(self._chunk_size) \
                    if self.shuffle else np.arange(self._chunk_size)
                index_array += (self.chunk_index * self._chunk_size)
                for self._step in range(self._chunk_size // self.batch_size):
                    self._total_batches_seen += 1
                    yield index_array[self._step * self.batch_size:(self._step + 1) * self.batch_size]

    def _next(self):
        with self._lock:
            index_array = next(self._index_generator)
            index_array.sort()
            out = [d[index_array] for d in self._data_in], [d[index_array] for d in self._data_out]
            out[0][0] = self.add_position(out[0][0])
            if self.get_vol: out[0][-1] = self.bound2vol(out[0][-1])
            out[0][-1] = self.add_position_bou(out[0][-1])
            if self.bypass: out[-1][0] = np.concatenate([out[-1][0], np.zeros_like(out[0][0])], -1)
        return out

    def __iter__(self):
        return self

    def __next__(self):
        return self._next()


class DIV2KDatasetReg(DIV2KDataset):
    def __init__(self, data_path,
                 mode='train',
                 block_shape='4x4',
                 batch_size=32,
                 chunk_size=30000,
                 hint=0,
                 samples=0,
                 shuffle=False,
                 get_vol=False,
                 bypass=False,
                 seed=42):
        super(DIV2KDatasetReg, self).__init__(data_path,
                                              mode=mode,
                                              block_shape=block_shape,
                                              batch_size=batch_size,
                                              chunk_size=chunk_size,
                                              hint=hint,
                                              samples=samples,
                                              shuffle=shuffle,
                                              get_vol=get_vol,
                                              bypass=bypass,
                                              seed=seed)

    def _next(self):
        with self._lock:
            index_array = next(self._index_generator)
            index_array.sort()
            out = [d[index_array] for d in self._data_in]
            if self.get_vol: out[-1] = self.bound2vol(out[-1])
            out += [d[index_array] for d in self._data_out]
        return out, []


class DIV2KDatasetRegMultiple:
    def __init__(self, data_path,
                 block_shape,
                 mode='train',
                 batch_size=32,
                 chunk_size=30000,
                 hint=0,
                 samples=0,
                 shuffle=False,
                 get_vol=False,
                 bypass=False,
                 seed=42):
        self.datasets = [DIV2KDatasetReg(data_path,
                                         mode=mode,
                                         block_shape=b,
                                         batch_size=batch_size,
                                         chunk_size=chunk_size,
                                         hint=hint,
                                         samples=samples,
                                         shuffle=shuffle,
                                         get_vol=get_vol,
                                         bypass=bypass,
                                         seed=seed) for b in block_shape]
        self.batch_size = batch_size
        self.samples = min([d.samples for d in self.datasets])
        if samples < self.samples and samples != 0: self.samples = samples
        self.nb_steps = self.samples // self.batch_size
        self._lock = threading.Lock()
        self._datagen = self._flow_multi()

    def set_batch_size(self, batch_size):
        for d in self.datasets:
            d.batch_size = batch_size

    def restart(self):
        for d in self.datasets:
            d.restart()

    def _flow_multi(self):
        while True:
            for _ in range(self.nb_steps):
                for d in self.datasets:
                    yield next(d)
            self.restart()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            batch = next(self._datagen)
            return batch


class DIV2KDatasetMultiple:
    def __init__(self, data_path,
                 block_shape,
                 mode='train',
                 batch_size=32,
                 chunk_size=30000,
                 samples=0,
                 shuffle=False,
                 get_vol=False,
                 distillate=False,
                 seed=42):
        self.datasets = [DIV2KDataset(data_path,
                                      mode=mode,
                                      block_shape=b,
                                      batch_size=batch_size,
                                      chunk_size=chunk_size,
                                      shuffle=shuffle,
                                      get_vol=get_vol,
                                      seed=seed) for b in block_shape]
        self.batch_size = batch_size
        self.samples = min([d.samples for d in self.datasets])
        if samples < self.samples and samples != 0: self.samples = samples
        self.nb_steps = self.samples // self.batch_size
        self._lock = threading.Lock()
        self._datagen = self._flow_multi()
        self.distillate = distillate

    def set_batch_size(self, batch_size):
        for d in self.datasets:
            d.batch_size = batch_size

    def restart(self):
        for d in self.datasets:
            d.restart()

    def _flow_multi(self):
        while True:
            for _ in range(self.nb_steps):
                for d in self.datasets:
                    yield next(d)
            self.restart()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            batch = next(self._datagen)
            return batch
