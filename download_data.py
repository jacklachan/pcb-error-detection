"""
Download the DeepPCB dataset.
Source: https://github.com/tangsanli5201/DeepPCB

Usage:
    python download_data.py
    python download_data.py --dest datasets/DeepPCB
"""
import argparse
import os
import subprocess
import sys


def download(dest):
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)

    if os.path.exists(dest):
        print(f'Directory {dest} already exists. Skipping download.')
        return

    print('Cloning DeepPCB dataset from GitHub...')
    result = subprocess.run(
        ['git', 'clone', 'https://github.com/tangsanli5201/DeepPCB', dest],
        capture_output=False,
    )

    if result.returncode != 0:
        print('\nGit clone failed. Try downloading manually:')
        print('  https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip')
        sys.exit(1)

    print(f'\nDataset downloaded to: {dest}')
    print('\nNext steps:')
    print(f'  python train.py --data_root {dest}')
    print(f'  python evaluate.py --data_root {dest} --checkpoint output/best_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', default='DeepPCB',
                        help='Where to clone the dataset (default: ./DeepPCB)')
    args = parser.parse_args()
    download(args.dest)
