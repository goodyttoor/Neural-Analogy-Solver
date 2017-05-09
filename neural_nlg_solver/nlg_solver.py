#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from random import random

import keras
import numpy as np
from scipy import ndimage
from scipy.misc import imresize

from neural_nlg_solver.AnalogyAlignment import AnalogyAlignment
from neural_nlg_solver.AnalogyNeuralModel import FullyConnectedModel
from neural_nlg_solver.nlg_generator import gen_fixed_len_nlgs

# Hide tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################################################################

__author__ = 'KAVEETA Vivatchai <vivatchai@fuji.waseda.jp, goodytong@gmail.com>'

# __date__, __version__ = '28/03/2017', '0.1'   # Generation 4, on-the-fly training ✈️
# __date__, __version__ = '05/04/2017', '0.2'   # Fix memory leak, stupid mistake!
# __date__, __version__ = '06/04/2017', '0.3'   # Workaround for Keras model loading bug
# __date__, __version__ = '13/04/2017', '0.4'   # Reuse generated training set
# __date__, __version__ = '01/05/2017', '0.5'   # Limit training base analogies by string lengths
# __date__, __version__ = '02/05/2017', '0.6'   # Change epoch strategy. Introduce min & max epochs and target loss
# __date__, __version__ = '03/05/2017', '0.7'   # Much simple decoder scheme
# __date__, __version__ = '05/05/2017', '0.8'   # Disable generation mode
# __date__, __version__ = '06/05/2017', '0.9'   # Single model mode
# __date__, __version__ = '07/05/2017', '0.10'  # Port first design to this script (fixed length single model)
__date__, __version__ = '08/05/2017', '1.0'     # Final ver, yeah i know really FINAL .... maybe until bugs found

__description__ = 'Train and experiments neural network model for solving proportional analogy equations'


########################################################################################################################

# Default model parameters
default_config = '{"optimizer": "Adam", "loss": "Binary_Crossentropy", "activation": "Sigmoid", "epochs": 1000, ' \
              '"batch": 64, "layers": []}'

# Base equation for model training
default_base = 'a : a :: b : b\n' \
               'a : a :: a : a\n' \
               'amo : oro :: amas : oras\n' \
               'dues : indu :: nées : inné\n' \
               'amo : oro :: amamus : oramus\n' \
               'singen : singt :: hören : hört\n' \
               'aslama : arsala :: muslim : mursil\n' \
               'honor : orator :: honorem : oratorem'


########################################################################################################################

def read_argv():
    """
    Read program arguments
    """

    program = 'v%s (c) %s %s' % (__version__, __date__.split('/')[2], __author__)
    description = __description__
    parser = argparse.ArgumentParser(prog=program, description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Test argument
    parser.add_argument('-i', '--in', action='store', dest='test_in', required=True, help='testing input path')
    parser.add_argument('-o', '--out', action='store', dest='test_out', required=True, help='testing output path')
    parser.add_argument('-md', '--model-dir', action='store', dest='model_dir', default='./model', help='model dir')

    # Model arguments
    parser.add_argument('-c', '--config', action='store', dest='config', default=None,
                        help='model configuration file (.cfg)')
    parser.add_argument('--design', action='store', dest='design', default='single',
                        choices=['single', 'multi'], help='neural network design:\n'
                                                          '   single: single fixed length model'
                                                          '   multi: multiple variable length models')
    parser.add_argument('--mode', action='store', dest='mode', default='bilinear',
                        choices=['append', 'gen', 'nearest', 'bilinear', 'bicubic'],
                        help='matrix interpolation mode:\n'
                             '   append: naively append with blank spaces\n'
                             '   gen: analogy generation\n'
                             '   nearest: nearest neighbour interpolation\n'
                             '   bilinear: bilinear interpolation\n'
                             '   bicubic: bicubic interpolation')
    parser.add_argument('--model_dim', action='store', dest='model_dim', type=int, default=-1,
                        help='dimension (length) of model (only for single mode, ignored on multi mode)\n'
                             '   -1: automatic select from train set')
    parser.add_argument('--filter', action='store', dest='filter', default=[], nargs='+',
                        choices=['morph', 'weight'], help='matrix filtration(s)\n'
                                                          '   morph: mathematical morphology\n'
                                                          '   weight: diagonal weighting')

    # Analogy generation arguments
    parser.add_argument('--offset', action='store', type=int, dest='offset', default=5,
                        help='string length offset for analogy generation mode [-1 to disable]')
    parser.add_argument('--min_gen', action='store', type=int, dest='min_gen', default=0,
                        help='minimum number of generated training samples')

    # Training arguments
    parser.add_argument('-tr', '--train', action='store', dest='train', default=None, help='training analogies file')
    parser.add_argument('--shuffle', action='store_true', dest='train_shf', help='shuffle training')

    parser.add_argument('--min_delta', action='store', type=float, dest='min_delta', default=0.00005,
                        help='early stopping minimum delta, minimum different learning rate between epoch to stop')
    parser.add_argument('--patience', action='store', type=int, dest='patience', default=3,
                        help='early stopping patience, number of epoch with lower delta learning rate than [min_delta] '
                             'before stop')

    # Test arguments
    parser.add_argument('--no_decode', action='store_true', dest='no_decode',
                        help='disable string decoding, only predict matrices')

    # Miscellaneous arguments
    parser.add_argument('-v', '--verbose', action='store', type=int, dest='verbose', default=0, choices=[0, 1, 2],
                        help='verbose level:\n   0: disable (quiet)\n   1: important messages\n   2: all messages')

    return parser.parse_args()


########################################################################################################################

def make_sure_path_exists(path):
    """
    Create directory if path not existed
    """
    if len(path) > 0:
        os.makedirs(path, exist_ok=True)


def load_data_set(data_path):
    """
    Load input set from path
    :param data_path: file to load analogies
    :return: list of AlignmentMatrices objects
    """

    data_set = []
    if data_path is not None:
        if '.gz' in data_path:
            txt_file = gzip.open(data_path, 'rt', encoding='utf8')
        else:
            txt_file = open(data_path, 'r', encoding='utf8')
        data_set = [AnalogyAlignment(line) for line in txt_file.read().split('\n') if len(line) > 0]

    return data_set


def save_data_set(data_set, save_path):
    """
    Save input set to path
    :param data_set: list of AlignmentMatrices objects
    :param save_path: file to save analogies
    """

    # Save generated analogies for future usage
    if '.gz' in save_path:
        out_file = gzip.open(save_path, 'wt', encoding='utf8')
    else:
        out_file = open(save_path, 'w', encoding='utf8')

    for nlg in data_set:
        print(nlg, file=out_file)

    out_file.close()


########################################################################################################################

def limit_train_set(train_set, str_lens, offset):
    """
    Limit number of analogies in training set by considering target length
    :param train_set: all training samples
    :param str_lens: target string lengths
    :param offset: string length offset
    :return: limited training set
    """
    limited_train_set = []

    # Get minimum target string length
    str_lens = sorted(str_lens)

    for train_nlg in train_set:
        tmp_str_lens = sorted([len(train_nlg.nlg[i][j]) for i, j in product([0, 1], [0, 1])])
        if all(str_lens[i] >= tmp_str_lens[i] >= str_lens[i] - offset for i in range(4)):
            limited_train_set.append(train_nlg)

    return limited_train_set


def gen_train_set(data_path, base_nlgs, str_lens):
    """
    Load training input set of specific string length from path
    :param data_path: file to load analogies
    :param base_nlgs: base analogies
    :param str_lens: string lengths (|A|, |B|, |C|, |D|)
    :return: list of AlignmentMatrices objects
    """

    # Read custom set
    gen_path = data_path.replace('.nlg', '_{}_{}_{}_{}.nlg'.format(*str_lens))

    if os.path.exists(gen_path):  # Load generated training set
        if args.verbose >= 1:
            print('Generated train set existed')

    else:  # Generate from base analogies
        if args.verbose >= 1:
            print('Generate train set for {}_{}_{}_{} from base'.format(*str_lens))

        # Determine offset value (-1: disable)
        gen_set = []
        if args.offset == -1:
            offset = sys.float_info.max
        else:
            offset = args.offset

        # Analogy generation
        while len(gen_set) < args.min_gen:
            gen_set = gen_fixed_len_nlgs([sample.nlg for sample in limit_train_set(base_nlgs, str_lens, offset)],
                                         str_lens)
            offset += 1

            # Break out when impossible to generate more
            if max(str_lens) - offset < 0:
                break

        # Save generated analogies for future usage
        save_data_set(gen_set, gen_path)

    return load_data_set(gen_path)


########################################################################################################################

def load_or_create_model(train_set, str_lens):
    """
    Load in trained model or create a new one
    :param train_set
    :param str_lens: string lengths (|A|, |B|, |C|, |D|)
    """

    # New model
    model = FullyConnectedModel()
    model_path = args.model_dir + '/' + args.design + '_' + args.mode + '_' + \
                                  '_'.join([str(str_len) for str_len in str_lens]) + '.h5'
    log_path = model_path.replace('.h5', '.log')

    if os.path.isfile(model_path):  # Model found, Re-load and use it to solve
        model.config(str_lens, params['layers'], params['optimizer'], params['loss'], params['activation'], False,
                     False)
        model.load(model_path)

    else:  # Not found, train a new model
        if args.verbose >= 1:
            print('Train new model: {} {} {} {}'.format(*str_lens))

        # Initiate a new model
        model.config(str_lens, params['layers'], params['optimizer'], params['loss'], params['activation'], True, False)

        train_gen_set = []
        if args.mode == 'gen':
            if args.verbose >= 1:
                print('Pre-process: Analogy generation')

            # Generate base equation to fixed-length analogies
            train_gen_set = gen_train_set(args.train, train_set, str_lens)

            # Fallback to default set if training set empty
            if len(train_gen_set) == 0:
                default_train_set = [AnalogyAlignment(line) for line in default_base.split('\n') if len(line) > 0]
                train_gen_set = gen_train_set(args.train, default_train_set, str_lens)

        elif args.mode == 'append' or args.mode == 'nearest' or args.mode == 'bilinear' or args.mode == 'bicubic':
            if args.verbose >= 1:
                print('Pre-process: {}'.format(args.mode.capitalize()))

            # Get all analogy shorter than current string lengths
            train_gen_set = [nlg for nlg in train_set if all(len(nlg.strs[i]) <= str_lens[i] for i in range(4))]

            # Fallback to default set if training set empty
            if len(train_gen_set) == 0:
                default_train_set = [AnalogyAlignment(line) for line in default_base.split('\n') if len(line) > 0]
                train_gen_set = [nlg for nlg in default_train_set if all(len(nlg.strs[i]) <= str_lens[i]
                                                                         for i in range(4))]

            # Interpolate to model dims
            for i in range(len(train_gen_set)):
                train_gen_set[i].alg_mats = interpolate_mat(train_gen_set[i].mats, str_lens, args.mode)

            # Optional matrix filtration
            for mat_filter in args.filter:
                if args.verbose >= 1:
                    print('Filter: {}'.format(mat_filter))

                # Apply filter
                if mat_filter == 'morph':
                    train_gen_set = filter_math_morph(train_gen_set)
                elif mat_filter == 'weight':
                    train_gen_set = filter_diagonal_weight(train_gen_set)

        if args.verbose >= 1:
            print('Number of processed training samples: {}'.format(len(train_gen_set)))

        # Train model
        model, history = train_model(model, params['epochs'], params['batch'], train_gen_set)

        # Save train history
        with open(log_path, 'wt') as log_file:
            json.dump(history, log_file)

        # Save model
        model.save(model_path)

        # Trigger garbage collection
        del train_gen_set

    return model


########################################################################################################################

def interpolate_mat(alg_mats, str_lens, mode):
    """
    Interpolate (rescale) the alignment matrices into target size determined by string lengths
    :param alg_mats: source alignment matrices (list of 4 numpy ndarray)
    :param str_lens: target string length
    :param mode: interpolation mode [append, nearest, bilinear, bicubic]
    :return: interpolated matrices
    """

    # Create new matrices with target dimensions
    new_mats = [np.zeros((str_lens[0], str_lens[1])), np.zeros((str_lens[0], str_lens[2])),
                np.zeros((str_lens[3], str_lens[1])), np.zeros((str_lens[3], str_lens[2]))]

    if mode == 'append':

        # Copy input from source matrices
        for i in range(4):

            row_num = min(alg_mats[i].shape[0], new_mats[i].shape[0])
            col_num = min(alg_mats[i].shape[1], new_mats[i].shape[1])

            # In case appending
            new_row_offset = 0
            new_col_offset = 0

            if i == 0 or i == 1:
                new_row_offset = new_mats[i].shape[0] - row_num
            if i == 0 or i == 2:
                new_col_offset = new_mats[i].shape[1] - col_num

            # In case clipping
            src_row_offset = 0
            src_col_offset = 0

            if i == 0 or i == 1:
                src_row_offset = alg_mats[i].shape[0] - row_num
            if i == 0 or i == 2:
                src_col_offset = alg_mats[i].shape[1] - col_num

            # Append to new matrices (DO clip the larger source matrices which should not be used)
            new_mats[i][new_row_offset:new_row_offset+row_num, new_col_offset:new_col_offset+col_num] = \
                alg_mats[i][src_row_offset:src_row_offset+row_num, src_col_offset:src_col_offset+col_num]

    elif mode == 'nearest' or mode == 'bilinear' or mode == 'bicubic':

        # Calculate scaling factor (minimum proportion)
        scale_factor = sys.float_info.max
        for i in range(4):
            scale_factor = min(scale_factor, new_mats[i].shape[0] / alg_mats[i].shape[0],
                               new_mats[i].shape[1] / alg_mats[i].shape[1])

        # Scaling
        for i in range(4):
            row_num = int(math.ceil(alg_mats[i].shape[0] * scale_factor))
            col_num = int(math.ceil(alg_mats[i].shape[1] * scale_factor))

            # In case appending
            new_row_offset = 0
            new_col_offset = 0

            if i == 0 or i == 1:
                new_row_offset = new_mats[i].shape[0] - row_num
            if i == 0 or i == 2:
                new_col_offset = new_mats[i].shape[1] - col_num

            new_mats[i][new_row_offset:new_row_offset + row_num, new_col_offset:new_col_offset + col_num] = \
                imresize(alg_mats[i], (row_num, col_num), interp=args.mode) / 255

    return new_mats


def filter_math_morph(nlgs):
    """ Perform mathematical morphology on input matrix
    :param nlgs: source AnalogyAlignment(s)
    :return: filtered matrix
    """

    # Create 2 filter elements
    elem_1 = np.zeros((3, 3))
    elem_2 = np.zeros((3, 3))
    for num in range(2):
        elem_1[num, num] = 1
        elem_2[-1 - num, -1 - num] = 1

    # Apply filters to all analogies
    for nlg_id in range(len(nlgs)):
        for mat_id in range(4):
            # Rotate to correct orientation
            nlgs[nlg_id].mats[mat_id] = np.rot90(nlgs[nlg_id].mats[mat_id], k=mat_id)

            # Filtering
            fil_mat_1 = ndimage.grey_erosion(nlgs[nlg_id].mats[mat_id], size=(3, 3), footprint=elem_1)
            fil_mat_2 = ndimage.grey_erosion(nlgs[nlg_id].mats[mat_id], size=(3, 3), footprint=elem_2)

            # Rotate back to original orientation
            nlgs[nlg_id].mats[mat_id] = np.rot90(np.maximum(fil_mat_1, fil_mat_2), k=4-mat_id)

    return nlgs


def filter_diagonal_weight(nlgs):
    """ Perform diagonal weighting on input matrix
    :param nlgs: source alignment matrices (list of 4 numpy ndarray)
    :return: filtered matrix
    """

    for nlg_id in range(len(nlgs)):
        for mat_id in range(4):
            # Rotate to correct orientation
            nlgs[nlg_id].mats[mat_id] = np.rot90(nlgs[nlg_id].mats[mat_id], k=mat_id)

            # Calculate diagonal distance
            dist = min(nlgs[nlg_id].mats[mat_id].shape) * math.sqrt(2) / 2

            for x_idx in range(nlgs[nlg_id].mats[mat_id].shape[0]):
                for y_idx in range(nlgs[nlg_id].mats[mat_id].shape[1]):
                    cell_dist = math.sqrt(2 * math.pow(abs(x_idx - y_idx), 2)) / 2

                    if cell_dist < dist:
                        nlgs[nlg_id].mats[mat_id][x_idx, y_idx] *= (1 - (cell_dist / dist))

            # Rotate back to original orientation
            nlgs[nlg_id].mats[mat_id] = np.rot90(nlgs[nlg_id].mats[mat_id], k=4-mat_id)

    return nlgs


def strip_pre_suffix(test_set):
    """
    Strip prefixes and suffixes from experiments sample (for decode single model + analogy generation model)
    :param test_set: list of AnalogyAlignment object(s)
    :return: stripped AnalogyAlignment object(s)
    """

    for test_id in range(len(test_set)):
        # Count prefix and suffix occurrences
        pre_nums = [test_set[test_id].nlg[int(str_id / 2)][str_id % 2].count('<') for str_id in range(4)]
        suf_nums = [test_set[test_id].nlg[int(str_id / 2)][str_id % 2].count('>') for str_id in range(4)]

        # Strip prefix and suffix from strings
        for str_id in range(4):
            i = int(str_id / 2)
            j = str_id % 2
            test_set[test_id].nlg[i][j] = test_set[test_id].nlg[i][j][
                pre_nums[str_id]:len(test_set[test_id].nlg[i][j])-suf_nums[str_id]]

        # Clip alignment matrices (only BD, CD)
        for mat_id in range(0, 4):

            row_offset_top = 0
            row_offset_bottom = 0
            col_offset_left = 0
            col_offset_right = 0

            # Get clipping offsets from strings
            if mat_id == 0 or mat_id == 1:
                row_offset_top = suf_nums[0]
                row_offset_bottom = pre_nums[0]

            if mat_id == 2 or mat_id == 3:
                row_offset_top = pre_nums[3]
                row_offset_bottom = suf_nums[3]

            if mat_id == 0 or mat_id == 2:
                col_offset_left = suf_nums[1]
                col_offset_right = pre_nums[1]

            if mat_id == 1 or mat_id == 3:
                col_offset_left = pre_nums[2]
                col_offset_right = suf_nums[2]

            tmp_mat = test_set[test_id].mats[mat_id]
            test_set[test_id].alg_mats[mat_id] = tmp_mat[
                row_offset_top:tmp_mat.shape[0]-row_offset_bottom,
                col_offset_left:tmp_mat.shape[1]-col_offset_right]

            del tmp_mat

    return test_set


def merge_nlgs(test_set):
    """
    Merge AnalogyAlignment objects. All alignment matrices must be the EXACT same size. Assume string from first object
    This method will be used by decoder on [single model + analogy generation]
    :param test_set: list of AnalogyAlignment object(s)
    :return: merge output as an AnalogyAlignment object
    """

    # Create a new AnalogyAlignment
    merged_test = test_set[0]

    # Zero out all alignment matrices
    for i in range(2, 4):
        merged_test.alg_mats[i] = np.zeros(merged_test.alg_mats[i].shape)

        # Sum
        for test_sample in test_set:
            merged_test.alg_mats[i] += test_sample.alg_mats[i]

        # Normalize
        merged_test.alg_mats[i] /= len(test_set)

    return merged_test


########################################################################################################################


def gen_src_trg(alg_mats):
    """
    Generate list of source and target from alignment matrices for training network model
    :param alg_mats: list of alignment matrices
    :return: list of source and target
    """

    # Split into source and target lists
    src = []
    trg = []
    for alg in alg_mats:
        src.append(np.concatenate([alg.mats[0], alg.mats[1]], axis=1))
        trg.append(np.concatenate([alg.mats[2], alg.mats[3]], axis=1))

    # Convert to ndarray
    src = np.array(src)
    trg = np.array(trg)

    return src, trg


########################################################################################################################

def train_model(model, epochs, batch, train_set):
    if args.verbose >= 1:
        print('Start model training')

    # Create train source and target
    train_src, train_trg = gen_src_trg(train_set)

    # Early stopping function
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=args.min_delta, patience=args.patience,
                                                   verbose=2 if args.verbose else 0, mode='auto')

    # Train
    train_result = model.train(train_src, train_trg, batch_size=batch, epochs=epochs, verbose=2 if args.verbose else 0,
                               callbacks=[early_stopping], validation_split=0.1, shuffle=args.train_shf)

    return model, {'train_losses': train_result.history['loss']}


def decode(test_sample, test_str_lens):
    """
    Decode analogy solution (string D) by looking at alignment matrices
    :param test_sample: an AnalogyAlignment object
    :param test_str_lens: list of string lengths [|A|, |B|, |C|, |D|]
    :return: decoded AnalogyAlignment object
    """

    # Decode
    # All characters
    chr_lst = list((Counter(list(test_sample.nlg[0][1])) + Counter(list(test_sample.nlg[1][0])) -
                    Counter(list(test_sample.nlg[0][0]))).elements())

    # Predict one character at a time
    sol_best = []
    for d_chr_id in range(test_str_lens[3]):

        # Calculate values for possible characters
        char_val = defaultdict(float)
        for char in list(set(chr_lst)):
            for b_chr_id in range(test_str_lens[1]):
                if char == test_sample.nlg[0][1][b_chr_id]:
                    char_val[char] = max(test_sample.alg_mats[2][d_chr_id, test_str_lens[1] - b_chr_id - 1],
                                         char_val[char])
            for c_chr_id in range(test_str_lens[2]):
                if char == test_sample.nlg[1][0][c_chr_id]:
                    char_val[char] = max(test_sample.alg_mats[3][d_chr_id, c_chr_id], char_val[char])

        # Pick the best character for current position
        best_chr = max(char_val, key=char_val.get)

        # More than one possible characters (very rare). Randomly pick either
        if len(best_chr) > 1:
            # Flip coin
            best_chr = random.choice(best_chr)

        # Remove this character from possible list
        chr_lst.remove(best_chr)

        # Add to solution
        sol_best.append(best_chr)

        del char_val, best_chr

    # Set best solution to experiments sample
    test_sample.nlg[1][1] = sol_best

    return test_sample


def solve_nlg(model, model_str_lens, test_set):
    """
    Solve an analogy equation
    :param model: trained model
    :param model_str_lens: model string length
    :param test_set: an AnalogyAlignment object
    :return: solved AnalogyAlignment objects
    """

    test_str_lens = [len(test_set.nlg[0][0]), len(test_set.nlg[0][1]),
                     len(test_set.nlg[1][0]), len(test_set.nlg[1][1])]

    # Interpolate to model size
    if args.design == 'single':
        if args.mode == 'gen':
            # Generate experiments set (casting only)
            test_set = [AnalogyAlignment(nlg_eq) for nlg_eq in gen_fixed_len_nlgs([test_set.nlg], model_str_lens,
                                                                                  methods='c')]

        elif args.mode == 'append' or args.mode == 'nearest' or args.mode == 'bilinear' or args.mode == 'bicubic':
            test_set.alg_mats = interpolate_mat(test_set.mats, model_str_lens, args.mode)
            test_set = [test_set]

    else:
        test_set = [test_set]

    # Optional matrix filtration
    for mat_filter in args.filter:
        if args.verbose >= 1:
            print('Filter: {}'.format(mat_filter))

        # Apply filter
        if mat_filter == 'morph':
            test_set = filter_math_morph(test_set)
        elif mat_filter == 'weight':
            test_set = filter_diagonal_weight(test_set)

    # Predict BD, CD
    test_src, _ = gen_src_trg(test_set)
    pred_mats = model.predict(test_src)

    # Split prediction matrix into left (BD), right (CD) matrices
    for i in range(len(test_set)):
        test_set[i].alg_mats[2] = pred_mats[i][:, 0:model_str_lens[1]]
        test_set[i].alg_mats[3] = pred_mats[i][:, model_str_lens[1]:]

    del model, test_src, pred_mats

    # Back to target experiments string length
    if args.design == 'single':
        if args.mode == 'gen':
            test_set = [merge_nlgs(strip_pre_suffix(test_set))]

        # Interpolate back
        elif args.mode == 'append' or args.mode == 'nearest' or args.mode == 'bilinear' or args.mode == 'bicubic':

            # Scale to square model which length = max(|A|, |B|, |C|, |D|)
            test_set[0].alg_mats = interpolate_mat(test_set[0].mats, [max(test_str_lens)] * 4, args.mode)

            # Clipping to experiments string lengths
            test_set[0].alg_mats = interpolate_mat(test_set[0].mats, test_str_lens, 'append')

    # At this point, there should be only one AnalogyAlignment object
    test_set = test_set[0]

    # Decode solution
    if not args.no_decode:
        test_set = decode(test_set, test_str_lens)

    # Print final solution
    if args.verbose >= 2:
        print('Solution: {} : {} :: {} : [{}]'.format(''.join(test_set.nlg[0][0]), ''.join(test_set.nlg[0][1]),
                                                      ''.join(test_set.nlg[1][0]), ''.join(test_set.nlg[1][1])))

    return test_set


########################################################################################################################

def main():
    # Create required directories
    make_sure_path_exists(args.model_dir)
    make_sure_path_exists(os.path.dirname(args.test_in))
    make_sure_path_exists(os.path.dirname(os.path.abspath(args.test_out)))

    # Fallback to default set. Save default set to model path
    if args.train is None:
        args.train = 'default.nlg.gz'
        with gzip.open(args.train, 'wt', encoding='utf8') as train_file:
            train_file.write(default_base)
        del train_file

    # Read base set
    train_set = load_data_set(args.train)
    if train_set and args.verbose >= 1:
        print('Training set read from: {}'.format(args.train))
        print('   Number of training sample(s): {}'.format(len(train_set)))

    # Remove temporary file
    if args.train == 'default.nlg.gz':
        os.remove('default.nlg.gz')

    # Read experiments set
    test_set = load_data_set(args.test_in)
    if test_set and args.verbose >= 1:
        print('Test set read from: {}'.format(args.test_in))
        print('   Number of testing sample(s): {}'.format(len(test_set)))

    # Temporary set D to "correct length" string
    for i in range(len(test_set)):
        test_set[i].nlg[1][1] = ['x' for _ in range(len(test_set[i].nlg[0][1]) + len(test_set[i].nlg[1][0]) -
                                                    len(test_set[i].nlg[0][0]))]

    # Open output stream
    if '.gz' in args.test_out:
        file_out = gzip.open(args.test_out, 'wt', encoding='utf8')
    else:
        file_out = open(args.test_out, 'w', encoding='utf8')

    # Initial placeholders
    model = None
    str_lens = None

    # Single model mode
    if args.design == 'single':
        if args.verbose >= 1:
            print('Design: Single fixed length model')

        # Calculate appropriate string lengths for our awesome single fixed length model
        if args.model_dim == -1:
            max_len = math.ceil(max(len(nlg.strs[i]) for i in range(4) for nlg in train_set) / 10) * 10
            str_lens = [max_len] * 4
        else:
            str_lens = [args.model_dim] * 4

        # Create or load model
        model = load_or_create_model(train_set, str_lens)

    else:
        if args.verbose >= 1:
            print('Design: Multiple variable length models')

    # Solve analogical equations
    for test_id in range(len(test_set)):
        if args.verbose >= 1:
            print('Solving {0} of {1}'.format(test_id + 1, len(test_set)))

        # If multi variable models, create model which match the current experiments sample string lengths
        if args.design == 'multi':

            # Get string lengths for current experiments sample
            str_lens = [len(test_set[test_id].nlg[0][0]), len(test_set[test_id].nlg[0][1]),
                        len(test_set[test_id].nlg[1][0]), len(test_set[test_id].nlg[0][1]) +
                        len(test_set[test_id].nlg[1][0]) - len(test_set[test_id].nlg[0][0])]

            # Get model for particular length
            model = load_or_create_model(train_set, str_lens)

        # Write solution to output file
        file_out.write(str(solve_nlg(model, str_lens, test_set[test_id])) + '\n')

    # Close file output stream
    file_out.close()


if __name__ == '__main__':

    # App start time
    start_time = time.time()

    # Read option arguments
    args = read_argv()

    # Sanity check
    if not os.path.exists(args.test_in):
        sys.exit('ERROR: Test input file not found')

    # Pre-load model configuration
    if args.config is not None:
        with open(args.config, 'r') as config_file:
            args.config = config_file.read()
    else:
        args.config = default_config

    params = json.loads(args.config)

    # Main process
    main()

    # Print execution time
    if args.verbose >= 1:
        print('Processing time: ' + ('%.2f' % (time.time() - start_time)) + 's')
