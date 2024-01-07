# Copyright 2021 SVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

from RCAN_master.rcan.callbacks import ModelCheckpoint, TqdmCallback
from RCAN_master.rcan.data_generator import DataGenerator
from RCAN_master.rcan.losses import mae, mse
from RCAN_master.rcan.metrics import psnr, ssim
from RCAN_master.rcan.model import build_rcan
from RCAN_master.rcan.utils import (
    convert_to_multi_gpu_model,
    get_gpu_count,
    normalize,
    staircase_exponential_decay)

import argparse
import itertools
import json
import jsonschema
import keras
import numpy as np
import pathlib
import tifffile
import os
# import cv2
import time
import random
import shutil


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def load_data(config, data_type):
    image_pair_list = config.get(data_type + '_image_pairs', [])

    if data_type + '_data_dir' in config:
        raw_dir, gt_dir = [
            pathlib.Path(config[data_type + '_data_dir'][t])
            for t in ['raw', 'gt']]

        raw_files, gt_files = [
            sorted(d.glob('*.tif')) for d in [raw_dir, gt_dir]]

        if not raw_files:
            raise RuntimeError('No TIFF file found in {raw_dir}')

        if len(raw_files) != len(gt_files):
            raise RuntimeError(
                '"{raw_dir}" and "{gt_dir}" must contain the same number of '
                'TIFF files')

        for raw_file, gt_file in zip(raw_files, gt_files):
            image_pair_list.append({'raw': str(raw_file), 'gt': str(gt_file)})

    if not image_pair_list:
        return None

    print('Loading {data_type} data')

    data = []
    for p in image_pair_list:
        raw_file, gt_file = [p[t] for t in ['raw', 'gt']]

        print('  - raw:', raw_file)
        print('    gt:', gt_file)

        raw, gt = [tifffile.imread(p[t]) for t in ['raw', 'gt']]

        if raw.shape != gt.shape:
            raise ValueError(
                'Raw and GT images must be the same size: '
                '{p["raw"]} {raw.shape} vs. {p["gt"]} {gt.shape}')

        data.append([normalize(m) for m in [raw, gt]])

    return data

# def load_data_custom(data_path, all_gt_img_data, all_noisy_data):
#     gt_img_path = data_path + '/' + 'gt_imgs'
#     noisy_img_path = data_path + '/' + 'noisy_imgs'

#     stack_list = os.listdir(gt_img_path)
#     stack_num = [int(stack[4:]) for stack in stack_list]
#     num_stacks = max(stack_num)

#     for stack in range(1, num_stacks + 1):
#         curr_img_gt = []
#         stack_path_gt = gt_img_path + '/img_' + str(stack)
#         curr_img_noisy = []
#         stack_path_noisy = noisy_img_path + '/img_' + str(stack)

#         zplane_list = os.listdir(stack_path_gt)
#         zplane_num = [int(zplane[2:len(zplane) - 4]) for zplane in zplane_list]
#         max_zplane_num = max(zplane_num)
#         for num in range(max_zplane_num):
#             curr_gt_z = cv2.imread(stack_path_gt + '/z_' + str(num + 1) + '.tif', -1)
#             # curr_img_gt.append(curr_gt_z[:, :, 0])
#             curr_img_gt.append(curr_gt_z)
#             curr_noisy_z = cv2.imread(stack_path_noisy + '/z_' + str(num + 1) + '.tif', -1)
#             # curr_img_noisy.append(curr_noisy_z[:, :, 0])
#             curr_img_noisy.append(curr_noisy_z)

#         curr_img_gt = np.array(curr_img_gt)
#         curr_img_noisy = np.array(curr_img_noisy)
#         curr_img_gt = np.transpose(curr_img_gt, axes=(1, 2, 0))
#         curr_img_noisy = np.transpose(curr_img_noisy, axes=(1, 2, 0))
#         curr_img_gt = np.amax(curr_img_gt, axis=2, keepdims= True)
#         curr_img_noisy = np.amax(curr_img_noisy, axis=2, keepdims= True)
#         all_gt_img_data.append(curr_img_gt)
#         all_noisy_data.append(curr_img_noisy)

#     return all_gt_img_data, all_noisy_data


# def get_depth_chunks_from_stack(img, depth):
#     depth_chunks_img = []
#     for z in range(img.shape[2]):
#         curr_image = []
#         below_frame_num = [n for n in range(int(z - (depth - 1) / 2), int(z))]
#         for below_frames in below_frame_num:
#             if below_frames < 0:
#                 curr_image.append(np.zeros((img.shape[0], img.shape[1])))
#             else:
#                 curr_image.append(img[:, :, below_frames])

#         curr_image.append(img[:, :, z])

#         above_frame_num = [n for n in range(int(z + 1), int(z + (depth - 1) / 2 + 1))]
#         for above_frames in above_frame_num:
#             if above_frames > img.shape[2] - 1:
#                 curr_image.append(np.zeros((img.shape[0], img.shape[1])))
#             else:
#                 curr_image.append(img[:, :, above_frames])

#         curr_image = np.array(curr_image)
#         curr_image = np.transpose(curr_image, axes=(1, 2, 0))
#         depth_chunks_img.append(curr_image)

#     return depth_chunks_img


# def prepare_training_data(all_gt_img_data, all_noisy_img_data, depth):
#     train_gt_img_data = []
#     train_noisy_img_data = []
#     for n in range(len(all_gt_img_data)):
#         curr_gt_img = all_gt_img_data[n]
#         curr_noisy_img = all_noisy_img_data[n]
#         depth_gt_img = get_depth_chunks_from_stack(curr_gt_img, 1)
#         depth_noisy_img = get_depth_chunks_from_stack(curr_noisy_img, depth)
#         train_gt_img_data.extend(depth_gt_img)
#         train_noisy_img_data.extend(depth_noisy_img)

#     train_gt_img_data = np.array(train_gt_img_data)
#     train_noisy_img_data = np.array(train_noisy_img_data)
#     return train_gt_img_data, train_noisy_img_data


def split_train_test(X, Y, split_ratio):
    idx = [i for i in range(X.shape[0])]
    random.shuffle(idx)
    test_size = round(X.shape[0]*split_ratio)
    train_X = X[idx[:X.shape[0] - test_size], :, :, :]
    train_Y = Y[idx[:X.shape[0] - test_size], :, :, :]
    test_X = X[idx[X.shape[0] - test_size:], :, :, :]
    test_Y = Y[idx[X.shape[0] - test_size:], :, :, :]

    return train_X, train_Y, test_X, test_Y


def data_to_rcan_format(train_gt_img_data, train_noisy_data):
    data = []
    for n in range(train_gt_img_data.shape[0]):
        data.append([normalize(train_noisy_data[n, :, :, 0]), normalize(train_gt_img_data[n, :, :, 0])])

    return data


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-o', '--output_dir', type=str, required=True)
args = parser.parse_args()

schema = {
    'type': 'object',
    'properties': {
        'training_image_pairs': {'$ref': '#/definitions/image_pairs'},
        'validation_image_pairs': {'$ref': '#/definitions/image_pairs'},
        'training_data_dir': {'$ref': '#/definitions/raw_gt_pair'},
        'validation_data_dir': {'$ref': '#/definitions/raw_gt_pair'},
        'input_shape': {
            'type': 'array',
            'items': {'type': 'integer', 'minimum': 1},
            'minItems': 2,
            'maxItems': 3
        },
        'num_channels': {'type': 'integer', 'minimum': 1},
        'num_residual_blocks': {'type': 'integer', 'minimum': 1},
        'num_residual_groups': {'type': 'integer', 'minimum': 1},
        'channel_reduction': {'type': 'integer', 'minimum': 1},
        'epochs': {'type': 'integer', 'minimum': 1},
        'steps_per_epoch': {'type': 'integer', 'minimum': 1},
        'data_augmentation': {'type': 'boolean'},
        'intensity_threshold': {'type': 'number'},
        'area_ratio_threshold': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'initial_learning_rate': {'type': 'number', 'minimum': 1e-6},
        'loss': {'type': 'string', 'enum': ['mae', 'mse']},
        'metrics': {
            'type': 'array',
            'items': {'type': 'string', 'enum': ['psnr', 'ssim']}
        }
    },
    'additionalProperties': False,
    'anyOf': [
        {'required': ['training_image_pairs']},
        {'required': ['training_data_dir']}
    ],
    'definitions': {
        'raw_gt_pair': {
            'type': 'object',
            'properties': {
                'raw': {'type': 'string'},
                'gt': {'type': 'string'},
            }
        },
        'image_pairs': {
            'type': 'array',
            'items': {'$ref': '#/definitions/raw_gt_pair'},
            'minItems': 1
        }
    }
}

with open(args.config) as f:
    config = json.load(f)

jsonschema.validate(config, schema)
config.setdefault('epochs', 300)
config.setdefault('steps_per_epoch', 256)
config.setdefault('num_channels', 32)
config.setdefault('num_residual_blocks', 3)
config.setdefault('num_residual_groups', 5)
config.setdefault('channel_reduction', 8)
config.setdefault('data_augmentation', True)
config.setdefault('intensity_threshold', 0.25)
config.setdefault('area_ratio_threshold', 0.5)
config.setdefault('initial_learning_rate', 1e-4)
config.setdefault('loss', 'mae')
config.setdefault('metrics', ['psnr'])

# training_data = load_data(config, 'training')
# validation_data = load_data(config, 'validation')

planaria_data = np.load('/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/CARE-dataset/Denoising_Planaria/train_data/data_label.npz')
# X = np.transpose(planaria_data['X'], (0, 2, 3, 4, 1))
# Y = np.transpose(planaria_data['Y'], (0, 2, 3, 4, 1))
# X = np.reshape(X, (-1, X.shape[2], X.shape[3], X.shape[4]))
# Y = np.reshape(Y, (-1, Y.shape[2], Y.shape[3], Y.shape[4]))
X = np.random.rand(1000, 64, 64, 1)
Y = np.random.rand(1000, 64, 64, 1)
train_X, train_Y, val_X, val_Y = split_train_test(X, Y, 0.2)
print(train_X.shape)
print(train_Y.shape)
print(val_X.shape)
print(val_Y.shape)

# # ######## custom data loader ##########
# # # base_data_path = ['D:/Shivesh/Denoising/20210226_denoising_ZIM504/cond_10ms110_10ms1000_v2']
# base_data_path = ['/storage/coda1/p-hl94/0/schaudhary9/testflight_data/denoising/20210626_denoising_OH16230_20x/cond_10ms75_10ms1000',
#                   '/storage/coda1/p-hl94/0/schaudhary9/testflight_data/denoising/20210627_denoising_OH16230_20x/cond_10ms75_10ms1000',
#                   '/storage/coda1/p-hl94/0/schaudhary9/testflight_data/denoising/20210630_denoising_OH16230_20x/cond_10ms75_10ms1000',
#                   '/storage/coda1/p-hl94/0/schaudhary9/testflight_data/denoising/20210705_denoising_OH16230_20x/cond_10ms75_10ms1000',
#                   '/storage/coda1/p-hl94/0/schaudhary9/testflight_data/denoising/20210718_denoising_OH16230_20x_0.75/cond_10ms75_10ms1000']

# # load data
# all_gt_img_data = []
# all_noisy_data = []
# for data_path in base_data_path:
#     all_gt_img_data, all_noisy_data = load_data_custom(data_path, all_gt_img_data, all_noisy_data)

# # prepare data for training
# train_gt_img_data, train_noisy_data = prepare_training_data(all_gt_img_data, all_noisy_data, 1)

# # split training and validation set
# # if tsize is not specified, split full data for train and test
# train_X, train_Y, test_X, test_Y = split_train_test(train_noisy_data, train_gt_img_data, 0.33)

# convert to RCAN data format
training_data = data_to_rcan_format(train_Y, train_X)
validation_data = None

ndim = training_data[0][0].ndim

for p in itertools.chain(training_data, validation_data or []):
    if p[0].ndim != ndim:
        raise ValueError('All images must have the same number of dimensions')

if 'input_shape' in config:
    input_shape = config['input_shape']
    if len(input_shape) != ndim:
        raise ValueError(
            '`input_shape` must be a {ndim}D array; received: {input_shape}')
else:
    input_shape = (16, 256, 256) if ndim == 3 else (256, 256)

for p in itertools.chain(training_data, validation_data or []):
    input_shape = np.minimum(input_shape, p[0].shape)

print('Building RCAN model')
print('  - input_shape =', input_shape)
for s in ['num_channels',
          'num_residual_blocks',
          'num_residual_groups',
          'channel_reduction']:
    print(f'  - {s} =', config[s])

model = build_rcan(
    (*input_shape, 1),
    num_channels=config['num_channels'],
    num_residual_blocks=config['num_residual_blocks'],
    num_residual_groups=config['num_residual_groups'],
    channel_reduction=config['channel_reduction'])

gpus = get_gpu_count()
model = convert_to_multi_gpu_model(model, gpus)

model.compile(
    optimizer=keras.optimizers.Adam(lr=config['initial_learning_rate']),
    loss={'mae': mae, 'mse': mse}[config['loss']],
    metrics=[{'psnr': psnr, 'ssim': ssim}[m] for m in config['metrics']])

data_gen = DataGenerator(
    input_shape,
    gpus,
    transform_function=(
        'rotate_and_flip' if config['data_augmentation'] else None),
    intensity_threshold=config['intensity_threshold'],
    area_ratio_threshold=config['area_ratio_threshold'])

training_data = data_gen.flow(*list(zip(*training_data)))

if validation_data is not None:
    validation_data = data_gen.flow(*list(zip(*validation_data)))
    checkpoint_filepath = 'weights_{epoch:03d}_{val_loss:.8f}.hdf5'
else:
    checkpoint_filepath = 'weights_{epoch:03d}_{loss:.8f}.hdf5'

if gpus != 0:
    steps_per_epoch = config['steps_per_epoch'] // gpus
else:
    steps_per_epoch = config['steps_per_epoch']
validation_steps = None if validation_data is None else steps_per_epoch

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


time_callback = TimeHistory()


print('Training RCAN model')
model.fit_generator(
    training_data,
    epochs=config['epochs'],
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_data,
    validation_steps=validation_steps,
    verbose=0,
    callbacks=[
        keras.callbacks.LearningRateScheduler(
            staircase_exponential_decay(config['epochs'] // 4)),
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir),
            write_graph=False),
        ModelCheckpoint(
            str(output_dir / checkpoint_filepath),
            monitor='loss' if validation_data is None else 'val_loss',
            save_best_only=True),
        TqdmCallback(),
        time_callback
    ])

a = 1