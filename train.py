import argparse
import os
from importlib.machinery import SourceFileLoader
import sys

from models.att_multi import m_norelu_down, m_norelu_pos
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True, help='config file')
parser.add_argument('-g', '--gpu', type=int, default=1, help='gpu number')
args = parser.parse_args()

cf = SourceFileLoader('config', args.config).load_module()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if cf.model == 'norelu_down':
    model = m_norelu_down.CrossIntraModel(cf)
elif cf.model == 'norelu_pos':
    model = m_norelu_pos.CrossIntraModel(cf)
else:
    raise ValueError('Invalid model')

model.train()
