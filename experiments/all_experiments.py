# REF: Adjustable parameters
#   1. Config
#       - Layers
#       - Loss
#       - Optimizer
#       - Dropout
#   2. Pipeline
#       - Single fixed length
#       - Multiple variable length
#   3. Mode
#       - Analogy generation
#           - Offset
#           - Min gen
#       - Append
#       - Nearest
#       - Bilinear
#       - Bicubic

import gzip
import json
import os

from collections import defaultdict

from neural_nlg_solver.AnalogyAlignment import AnalogyAlignment

# Directories
config_path = './config/'
model_path = './model/'
data_path = './input/'
result_path = './output/'

# Train and experiments sets
cross_num = 10
train_sets = ['combined_train_{}.nlg.gz'.format(i) for i in range(cross_num)]
test_sets = ['combined_test_{}.nlg.gz'.format(i) for i in range(cross_num)]

# Default parameters
default_config = 'config/5.cfg'
default_design = 'single'
default_mode = 'bilinear'
default_filter = 'morph weight'
default_model_dir = 16


# ======================================================================================================================

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


def cal_acc(truth_path, predict_path):
    """
    Calculate prediction accuracy by comparing predicted solutions to ground truth
    :param truth_path: path to ground truth file
    :param predict_path: path to predict file
    :return: accuracy [0, 100]
    """

    # Load data
    truth_data = load_data_set(truth_path)
    predict_data = load_data_set(predict_path)

    # Calculate accuracy
    correct_num = 0
    for nlg_id in range(min(len(truth_data), len(predict_data))):
        if truth_data[nlg_id].nlg[1][1] == predict_data[nlg_id].nlg[1][1]:
            correct_num += 1

    # Print result to stdout
    return 100 * correct_num / min(len(truth_data), len(predict_data))


# ======================================================================================================================

# Experiment 1: single vs multi
exp_name = 'exp1'

# for design in ['single', 'multi']:
for design in ['multi']:
    accuracies = defaultdict(list)

    for cross_id in range(cross_num):
        print('================================================')
        print('Cross validation set: {}'.format(cross_id))
        print('   Train set: {}'.format(train_sets[cross_id]))
        print('   Test set: {}'.format(test_sets[cross_id]))

        # Input files
        train_path = 'input/{}'.format(train_sets[cross_id])
        test_path = 'input/{}'.format(test_sets[cross_id])

        # Output files
        predict_path = 'output/{}/{}/{}.nlg.gz'.format(exp_name, design, cross_id)
        model_dir = 'model/{}/{}/{}/'.format(exp_name, design, cross_id)

        # Solve analogies
        os.system('python3 ../neural_nlg_solver/nlg_solver.py -i {} -o {} -tr {} -md {} -c {} '
                  '--design {} --mode {} --filter {} --model_dim {} -v 1'.format(
                    test_path, predict_path, train_path, model_dir, default_config,
                    design, default_mode, default_filter, default_model_dir))

        # Calculate accuracy
        accuracies[design].append(cal_acc(test_path, predict_path))

    # Save results
    with open(result_path + '{}_{}.json'.format(exp_name, design), 'w', encoding='utf8') as result_log:
        json.dump(accuracies, result_log)


# ======================================================================================================================

# Experiment 2: model configurations
exp_name = 'exp2'
configs = ['config/{}.cfg'.format(i) for i in [0, 5, 10, 15, 20]]

for config in configs:
    accuracies = defaultdict(list)

    for cross_id in range(cross_num):
        print('================================================')
        print('Cross validation set: {}'.format(cross_id))
        print('   Train set: {}'.format(train_sets[cross_id]))
        print('   Test set: {}'.format(test_sets[cross_id]))

        # Input files
        train_path = 'input/{}'.format(train_sets[cross_id])
        test_path = 'input/{}'.format(test_sets[cross_id])

        # Output files
        predict_path = 'output/{}/{}/{}.nlg.gz'.format(exp_name, os.path.basename(config).replace('.', '_'), cross_id)
        model_dir = 'model/{}/{}/{}/'.format(exp_name, os.path.basename(config).replace('.', '_'), cross_id)

        # Solve analogies
        os.system('python3 ../neural_nlg_solver/nlg_solver.py -i {} -o {} -tr {} -md {} -c {} '
                  '--design {} --mode {} --filter {} --model_dim {} -v 1'.format(
                    test_path, predict_path, train_path, model_dir, config,
                    default_design, default_mode, default_filter, default_model_dir))

        # Calculate accuracy
        accuracies[os.path.basename(config).replace('.', '_')].append(cal_acc(test_path, predict_path))

    # Save results
    with open(result_path + '{}_{}.json'.format(exp_name, os.path.basename(config).replace('.', '_')),
              'w', encoding='utf8') as result_log:
        json.dump(accuracies, result_log)


# ======================================================================================================================

# Experiment 3: Modes
exp_name = 'exp3'
modes = ['append', 'nearest', 'bilinear', 'bicubic', 'gen']

for mode in modes:
    accuracies = defaultdict(list)

    for cross_id in range(cross_num):
        print('================================================')
        print('Cross validation set: {}'.format(cross_id))
        print('   Train set: {}'.format(train_sets[cross_id]))
        print('   Test set: {}'.format(test_sets[cross_id]))

        # Input files
        train_path = 'input/{}'.format(train_sets[cross_id])
        test_path = 'input/{}'.format(test_sets[cross_id])

        # Output files
        predict_path = 'output/{}/{}/{}.nlg.gz'.format(exp_name, mode, cross_id)
        model_dir = 'model/{}/{}/{}/'.format(exp_name, mode, cross_id)

        # Solve analogies
        os.system('python3 ../neural_nlg_solver/nlg_solver.py -i {} -o {} -tr {} -md {} -c {} '
                  '--design {} --mode {} --filter {} --model_dim {} -v 1'.format(
                    test_path, predict_path, train_path, model_dir, default_config,
                    'multi', mode, default_filter, default_model_dir))

        # Calculate accuracy
        accuracies[mode].append(cal_acc(test_path, predict_path))

    # Save results
    with open(result_path + '{}_{}.json'.format(exp_name, mode), 'w', encoding='utf8') as result_log:
        json.dump(accuracies, result_log)
