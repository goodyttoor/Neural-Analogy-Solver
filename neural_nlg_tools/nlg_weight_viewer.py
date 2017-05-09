#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import json
import math
import os
import sys
from os.path import basename

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QFileDialog, QMessageBox, QTableWidgetItem, QAbstractItemView
from PyQt5.uic import loadUiType
from keras.layers.core import Dense

# Hide Tensorflow warning
from neural_nlg_solver.AnalogyNeuralModel import FullyConnectedModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Ui_MainWindow, QMainWindow = loadUiType('./interface/nlg_weight_viewer.ui')

###############################################################################

__author__ = 'KAVEETA Vivatchai <vivatchai@fuji.waseda.jp>'

# __date__, __version__ = '09/02/2017', '0.1'  # Prototype version
# __date__, __version__ = '10/02/2017', '0.2'  # First working version
# __date__, __version__ = '19/02/2017', '0.3'  # Mouse selection
# __date__, __version__ = '22/02/2017', '0.4'  # Coverage weight matrices
__date__, __version__ = '28/02/2017', '0.5'  # Non-uniform model support

__description__ = 'Visualize neural network model weight (Legacy)'


# FIXME: update to final design, postponed until further notice


###############################################################################

class VisualizerInterface(QMainWindow, Ui_MainWindow):
    def __init__(self):
        """
        Interface initialization
        """

        super(VisualizerInterface, self).__init__()
        self.setupUi(self)
        self.center()

        # Disable table edit, single selection
        self.tableAlignment.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableAlignment.setSelectionMode(QAbstractItemView.SingleSelection)

        # Action handlers
        # self.resizeEvent = self.action_resize
        self.actionOpen.triggered.connect(self.action_open)
        self.actionAbout.triggered.connect(self.action_about)
        self.actionExit.triggered.connect(self.close)
        self.actionZoomIn.triggered.connect(self.action_zoom_in)
        self.actionZoomOut.triggered.connect(self.action_zoom_out)
        self.actionZoomFit.triggered.connect(self.action_zoom_fit)
        self.actionNavFirst.triggered.connect(self.action_nav_first)
        self.actionNavLast.triggered.connect(self.action_nav_last)
        self.actionNavPrev.triggered.connect(self.action_nav_prev)
        self.actionNavNext.triggered.connect(self.action_nav_next)
        self.actionCopyTex.triggered.connect(self.action_copy_tex)
        self.spinBoxLayer.valueChanged.connect(self.action_spinbox_layer_value_changed)
        self.horizontalSlider.valueChanged.connect(self.action_slider_value_changed)
        # self.tableAlignment.selectionModel().currentChanged.connect(self.action_table_select_changed)

        # Default value
        # self.model = None
        self.weights = None
        self.shapes = []
        self.layer_num = 0
        self.cell_width = 8
        self.layer_id = 0
        self.cell_id = 0

        self.update_status('Ready to load alignment')

    def action_open(self):
        """
        Handler for open menu
        """

        # Read config file
        config_path = QFileDialog.getOpenFileName(self, 'Load network config', '.', 'Config file (*.cfg)')[0]

        if config_path is None:
            return

        with open(config_path, 'r') as config_file:
            params = json.loads(config_file.read())

        # Read weight file
        weight_file = QFileDialog.getOpenFileName(self, 'Load network weights', '.', 'Weight file (*.h5)')[0]

        if weight_file is None:
            return

        # Get string lengths from file name
        str_lens = [int(str_len) for str_len in os.path.splitext(basename(weight_file))[0].split('_')]

        # Initiate a new model
        model = FullyConnectedModel()
        model.config(str_lens, params['layers'], params['optimizer'], params['loss'], params['activation'],
                     False, False, False)

        # Load weights
        model.load(weight_file)

        self.weights = model.weights

        # Get layer shapes
        self.shapes = []

        # - Hidden layers
        for layer in model.layers:
            if type(layer) == Dense:
                self.shapes.append((1, layer.output_shape[1]))

        # - Output layer
        self.shapes[-1] = (str_lens[3], (str_lens[1] + str_lens[2]))
        # self.shapes.append(model.output_shape[1:3])

        # Set slider length
        # Layer numbers
        self.spinBoxLayer.setMaximum(int(len(self.weights) / 2) - 1)
        self.spinBoxLayer.setValue(0)

        # Cell numbers
        self.horizontalSlider.setMaximum(len(self.weights[0]) - 1)
        self.horizontalSlider.setValue(0)

        self.action_zoom_fit()

    @staticmethod
    def action_about():
        msg = QMessageBox()
        msg.setWindowTitle('About')
        msg.setText('Weight matrix visualizer\nVersion {}'.format(__version__))
        msg.setInformativeText(__author__)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setIconPixmap(QtGui.QPixmap('interface/avatar.png'))
        msg.exec_()

    def action_zoom_in(self):
        self.cell_width += 1
        self.update_table()

    def action_zoom_out(self):
        if self.cell_width > 1:
            self.cell_width -= 1
            self.update_table()

    def action_zoom_fit(self):
        if self.weights is None:
            return

        frame_geometry = self.tableAlignment.geometry()
        cell_width = (frame_geometry.width() - 5) / self.shapes[self.layer_id][1]
        cell_height = (frame_geometry.height() - 5) / (self.shapes[self.layer_id][0] * 2 + 1)

        self.cell_width = min(cell_width, cell_height)
        self.update_table()

    def action_spinbox_layer_value_changed(self):
        self.layer_id = self.spinBoxLayer.value()
        self.update_table()

    def action_slider_value_changed(self):
        self.cell_id = self.horizontalSlider.value()
        self.update_table()

    def action_nav_first(self):
        if self.weights:
            self.horizontalSlider.setValue(0)
            self.update_table()

    def action_nav_last(self):
        if self.weights:
            self.horizontalSlider.setValue(self.horizontalSlider.maximum())
            self.update_table()

    def action_nav_prev(self):
        if self.weights and self.cell_id > 0:
            self.horizontalSlider.setValue(self.cell_id - 1)
            self.update_table()

    def action_nav_next(self):
        if self.weights and self.cell_id <= self.horizontalSlider.maximum():
            self.horizontalSlider.setValue(self.cell_id + 1)
            self.update_table()

    @staticmethod
    def mat_to_tex_str(np_mat):
        """
        Convert matrix to latex array string literal
        :param np_mat: Numpy matrix or array
        :return: latex string
        """

        # Normalize to range [0,100]
        tmp_mat = np_mat.copy()
        mat_min = np.min(tmp_mat)
        mat_max = np.max(tmp_mat)
        tmp_mat = (tmp_mat - mat_min) * 100 / (mat_max - mat_min)

        np.set_printoptions(threshold=np.nan)
        return np.array2string(tmp_mat.astype(int), precision=2, suppress_small=True, separator=',',
                               max_line_width=sys.maxsize).replace('[', '').replace(' ', '').replace('\n', '')\
                               .replace('.,', ',').replace('.]', ']').replace('],', '\\\\').replace(']', '')

    def action_copy_tex(self):
        """
        Copy latex code for displaying alignment matrices to clipboard
        """

        # if self.weights:
        #
        #     # Get bias matrices
        #     bias_mat = self.weights[self.layer_id * 2 + 1]
        #     bias_dim = int(math.sqrt(len(bias_mat) / 2))
        #
        #     bias_mat_left = bias_mat[:int(len(bias_mat)/2)]
        #     bias_mat_left = np.reshape(bias_mat_left, (bias_dim, bias_dim))
        #
        #     bias_mat_right = bias_mat[int(len(bias_mat)/2):]
        #     bias_mat_right = np.reshape(bias_mat_right, (bias_dim, bias_dim))
        #
        #     tex_str = '% Bias matrices\n'
        #     tex_str += '\mat{{{0}}}\n\hspace{{1cm}}\n\mat{{{1}}}\n\\vspace{{1cm}}\n\\\\\n'.format(
        #         self.mat_to_tex_str(bias_mat_left),
        #         self.mat_to_tex_str(bias_mat_right))
        #
        #     # Get weight matrices
        #     wgh_mat = self.weights[self.layer_id * 2][self.cell_id]
        #     wgh_dim = int(math.sqrt(len(wgh_mat) / 2))
        #
        #     wgh_mat_left = wgh_mat[:int(len(wgh_mat)/2)]
        #     wgh_mat_left = np.reshape(wgh_mat_left, (wgh_dim, wgh_dim))
        #
        #     wgh_mat_right = wgh_mat[int(len(wgh_mat) / 2):]
        #     wgh_mat_right = np.reshape(wgh_mat_right, (wgh_dim, wgh_dim))
        #
        #     # Weight matrices (Currently selected cell)
        #     tex_str += '% Weight matrices #{}\n'.format(self.cell_id)
        #     tex_str += '\mat{{{0}}}\n\hspace{{1cm}}\n\mat{{{1}}}\n\\vspace{{1cm}}\n\\\\\n'.format(
        #         self.mat_to_tex_str(wgh_mat_left), self.mat_to_tex_str(wgh_mat_right))
        #
        #     # Coverage matrices
        #     tex_str += '% Coverage weight matrices left\n'.format(self.cell_id)
        #     tex_str += '\mat{{{0}}}\n\hspace{{1cm}}\n\mat{{{1}}}'.format(self.mat_to_tex_str(self.left_cover),
        #                                                                  self.mat_to_tex_str(self.right_cover))
        #
        #     # Set clipboard
        #     QApplication.clipboard().setText(tex_str)
        #
        #     self.statusBar.showMessage('Latex code copied to clipboard')

    def action_resize(self, _):
        """
        Handle app windows resize. Zoom table to fit new size
        """

        self.action_zoom_fit()

    def action_cover_box(self):
        """
        Handle toolbar cover box button click
        """

        self.update_table()

    def update_status(self, message):
        self.statusBar.showMessage(message)

    def center(self):
        """
        Snippet to move app window to center of screen
        """

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


###############################################################################

    @staticmethod
    def reset_table(table, cell_size):
        """
        Clear table view by set all cell to new item
        """
        # Set cell size
        for row in range(table.rowCount()):
            table.setRowHeight(row, cell_size)

        for col in range(table.columnCount()):
            table.setColumnWidth(row, cell_size)

        # Clear cell input
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                item = QTableWidgetItem()
                table.setItem(row, col, item)

    def update_table(self):
        """
        Update table widget
        """

        if self.weights is None or self.layer_id < 0:
            self.statusBar.showMessage('Warning: no model loaded')
            return

        # Bias mat dim
        bias_mat_dim = self.shapes[self.layer_id]

        # Weight mat dim
        weight_mat_dim = self.shapes[self.layer_id]

        # Set cell size
        self.tableAlignment.setColumnCount(max(bias_mat_dim[1], weight_mat_dim[1]))
        self.tableAlignment.setRowCount(bias_mat_dim[0] + weight_mat_dim[0] + 1)

        # Clear table
        self.reset_table(self.tableAlignment, self.cell_width)

        # Bias matrix
        bias_mat = self.weights[self.layer_id * 2 + 1]
        bias_min = min(bias_mat)
        bias_rng = max(bias_mat) - bias_min
        for cell_id in range(len(bias_mat)):
            cell_value = int(math.floor(255 * (bias_mat[cell_id] - bias_min) / bias_rng))
            item = QTableWidgetItem()
            item.setBackground(QtGui.QColor(cell_value, cell_value, cell_value))
            self.tableAlignment.setItem(int(cell_id / bias_mat_dim[1]), int(cell_id % bias_mat_dim[1]), item)

        # Weight matrix
        wgh_mat = self.weights[self.layer_id * 2][self.cell_id]
        wgh_min = min(wgh_mat)
        wgh_rng = max(wgh_mat) - wgh_min
        for cell_id in range(len(wgh_mat)):
            cell_value = int(math.floor(255 * (wgh_mat[cell_id] - wgh_min) / wgh_rng))
            item = QTableWidgetItem()
            item.setBackground(QtGui.QColor(cell_value, cell_value, cell_value))
            self.tableAlignment.setItem(int(cell_id / weight_mat_dim[1]) + weight_mat_dim[0] + 1,
                                        int(cell_id % weight_mat_dim[1]), item)

        self.update_status('Table updated. Cell #{} selected.'.format(self.cell_id + 1))


###############################################################################

def read_argv():
    program = 'v%s (c) %s %s' % (__version__, __date__.split('/')[2], __author__)
    parser = argparse.ArgumentParser(prog=program, description=__description__)
    return parser.parse_args()


def main():
    viz = VisualizerInterface()
    viz.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('interface/app.png'))
    args = read_argv()
    main()
