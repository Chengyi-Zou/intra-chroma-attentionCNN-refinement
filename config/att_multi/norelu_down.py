experiment_name = "base1"
output_path = "path/to/experiment"
data_path = "path/to/blocks"

# Model parameters
model = 'norelu_down'
block_shape = ['4x4', '8x8', '16x16']
bb1, bb2 = 32, 32
lb1, lb2 = 64, 64
tb = 32
att_h = 16
ext_bound = True
temperature = 0.5
multi_model = True

# Training parameters
epochs = 90
batch_size = 16
validate = True
shuffle = True
use_multiprocessing = False

# Optimizer
lr = 0.00001  # from 0.0001
beta = 0.9

# Early stop
es_patience = 15  # from 10
