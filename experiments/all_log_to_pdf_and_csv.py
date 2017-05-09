#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################

__author__ = 'KAVEETA Vivatchai <vivatchai@fuji.waseda.jp, goodytong@gmail.com>'

__date__, __version__ = '15/02/2017', '0.1'  # First version

__description__ = 'Convert all log files in directory to pdf and csv for latex plotting'

log_dir = './model/0/0/'


###############################################################################

def plot_graph(data):
    fig = plt.figure()
    plt.plot(data)
    return fig


###############################################################################

# Get all file list
log_files = []

for file in os.listdir(log_dir):
    if file.endswith(".log"):
        log_files.append(file)

# Create csv files
for file in log_files:

    with open(log_dir + file, encoding='utf8') as log_file:
        train_losses = json.loads(log_file.read())['train_losses']

    # Write plot to pdf
    pdf_plot = PdfPages(log_dir + file.replace('.log', '.pdf'))
    pdf_plot.savefig(plot_graph(train_losses))
    pdf_plot.close()

    # Write to csv
    with open(log_dir + file.replace('.log', '.csv'), 'w', encoding='utf8') as train_csv:
        # Write first header row
        train_csv.write('x,y\n')

        # Write input
        for loss_id, loss in enumerate(train_losses):
            train_csv.write('{},{}\n'.format(loss_id + 1, loss))
