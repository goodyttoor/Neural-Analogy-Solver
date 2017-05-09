# Generate test set by removing string D from analogical equations

import gzip

# 1. Read nlg file
from random import shuffle

input_file = 'combined.nlg.gz'

data_set = None
if input_file is not None:
    if '.gz' in input_file:
        txt_file = gzip.open(input_file, 'rt', encoding='utf8')
    else:
        txt_file = open(input_file, 'r', encoding='utf8')
    data_set = [line.replace(',', ';') for line in txt_file.read().split('\n') if len(line) > 0]

# 2. Split ten folds
shuffle(data_set)
data_chunks = list(zip(*[iter(data_set)]*(len(data_set)//10)))

# 3. Create new train & test sets
for cross_id in range(10):

    # Write train file
    train_path = input_file.replace('.nlg', '_train_{}.nlg'.format(cross_id))
    train_data = []
    [train_data.extend(data_chunks[tmp_id]) for tmp_id in range(10) if tmp_id == cross_id]

    if '.gz' in train_path:
        train_file = gzip.open(train_path, 'wt', encoding='utf8')
    else:
        train_file = open(train_path, 'w', encoding='utf8')

    train_file.write('\n'.join(train_data))
    train_file.close()

    # Write test file
    test_path = input_file.replace('.nlg', '_test_{}.nlg'.format(cross_id))
    test_data = []
    [test_data.extend(data_chunks[tmp_id]) for tmp_id in range(10) if tmp_id != cross_id]

    if '.gz' in test_path:
        test_file = gzip.open(test_path, 'wt', encoding='utf8')
    else:
        test_file = open(test_path, 'w', encoding='utf8')

    test_file.write('\n'.join(test_data))
    test_file.close()
