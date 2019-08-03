import argparse

parser = argparse.ArgumentParser()
parser.add_argument('arch')
parser.add_argument('ckpt')
parser.add_argument('gpu')
parser.add_argument('--height', default='256')
parser.add_argument('--dataset', default='market1501')

parsed = parser.parse_args()

import subprocess
subprocess.Popen(
    [
        'python', 'train_reg_crit.py',
        '-s', 'market1501',
        '-t', *parsed.dataset.split(','),
        '--height', parsed.height,
        '--width', '128',
        '--test-batch-size', '100',
        '--evaluate',
        '-a', parsed.arch,
        '--load-weights', parsed.ckpt,
        '--save-dir', '../__',
        '--gpu-devices', parsed.gpu
    ]).communicate()
