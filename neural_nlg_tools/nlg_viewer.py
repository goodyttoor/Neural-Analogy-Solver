#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import gzip
import itertools
import sys

from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QFileDialog, \
    QMessageBox, QTableWidgetItem
from PyQt5.uic import loadUiType

from neural_nlg_solver.AnalogyAlignment import AnalogyAlignment

Ui_MainWindow, QMainWindow = loadUiType('./interface/nlg_viewer.ui')

###############################################################################

__author__ = 'KAVEETA Vivatchai <vivatchai@fuji.waseda.jp, goodytong@gmail.com>'

# __date__, __version__ = '10/01/2017', '1.0'  # First version
# __date__, __version__ = '15/03/2017', '1.1'  # Latex export
# __date__, __version__ = '28/03/2017', '1.2'  # Non-uniform matrices support
__date__, __version__ = '06/04/2017', '1.3'  # Determine |D| from |B|+|C|-|A| for experiments set

__description__ = 'Visualize alignment matrices'


###############################################################################

class VisualizerInterface(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(VisualizerInterface, self).__init__()
        self.setupUi(self)
        self.center()
        self.resizeEvent = self.action_resize

        # Set action handlers
        self.actionOpen.triggered.connect(self.action_open)
        self.actionAbout.triggered.connect(self.action_about)
        self.actionExit.triggered.connect(self.close)
        self.actionZoomIn.triggered.connect(self.action_zoom_in)
        self.actionZoomOut.triggered.connect(self.action_zoom_out)
        self.actionZoomFit.triggered.connect(self.action_zoom_fit)
        self.actionTextBig.triggered.connect(self.action_text_big)
        self.actionTextSmall.triggered.connect(self.action_text_small)
        self.actionNavFirst.triggered.connect(self.action_nav_first)
        self.actionNavLast.triggered.connect(self.action_nav_last)
        self.actionNavPrev.triggered.connect(self.action_nav_prev)
        self.actionNavNext.triggered.connect(self.action_nav_next)
        self.actionEqualPoints.triggered.connect(self.action_equal_points)
        self.actionCopyTex.triggered.connect(self.action_copy_tex)
        self.horizontalSlider.valueChanged.connect(self.action_slider_value_changed)

        # Default value
        self.align_mats = None
        self.equal_char = '◯'
        self.align_id = 0
        self.cell_width = 8
        self.text_cell = 30
        self.update_status('Ready to load alignment')

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    @staticmethod
    def action_about():
        msg = QMessageBox()
        msg.setWindowTitle('About')
        msg.setText('Alignment matrix visualizer\nVersion {}'.format(__version__))
        msg.setInformativeText(__author__)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setIconPixmap(QtGui.QPixmap('interface/avatar.png'))
        msg.exec_()

    def action_open(self):
        input_file = QFileDialog.getOpenFileName(self, 'Load alignment', '.',
                                                 'Alignment files (*.nlg *.nlg.gz)')[0]
        if input_file:
            self.align_mats = read_input(input_file)
            self.horizontalSlider.setMaximum(len(self.align_mats) - 1)
            self.horizontalSlider.setValue(0)
            self.best_fit()
            self.update_table()

    def action_zoom_in(self):
        self.cell_width += 5
        self.update_table()

    def action_zoom_out(self):
        if self.cell_width > 5:
            self.cell_width -= 5
            self.update_table()

    def action_text_big(self):
        self.text_cell += 10
        self.best_fit()
        self.update_table()

    def action_text_small(self):
        if self.text_cell > 15:
            self.text_cell -= 10
            self.best_fit()
            self.update_table()

    def action_zoom_fit(self):
        self.best_fit()
        self.update_table()

    def action_slider_value_changed(self):
        self.align_id = self.horizontalSlider.value()
        self.best_fit()
        self.update_table()

    def action_nav_first(self):
        if self.align_mats is None:
            return

        self.horizontalSlider.setValue(0)

    def action_nav_last(self):
        if self.align_mats is None:
            return

        self.horizontalSlider.setValue(len(self.align_mats) - 1)

    def action_nav_prev(self):
        if self.align_mats and self.align_id > 0:
            self.horizontalSlider.setValue(self.align_id - 1)

    def action_nav_next(self):
        if self.align_mats and self.align_id < len(self.align_mats):
            self.horizontalSlider.setValue(self.align_id + 1)

    def action_equal_points(self):
        self.update_table()

    def action_copy_tex(self):
        if self.align_mats is None:
            return

        # Copy latex code to clipboard
        QApplication.clipboard().setText(self.align_mats[self.align_id].tex_str)
        self.statusBar.showMessage('Latex code copied to clipboard')

    def action_resize(self, _):
        self.best_fit()
        self.update_table()

    def best_fit(self):
        if self.align_mats is None:
            return

        str_a = self.align_mats[self.align_id].nlg[0][0]
        str_b = self.align_mats[self.align_id].nlg[0][1]
        str_c = self.align_mats[self.align_id].nlg[1][0]

        col_num = len(str_b) + len(str_c)
        row_num = len(str_a) + (len(str_b) + len(str_c) - len(str_a))

        frame_geometry = self.tableAlignment.geometry()
        cell_width = (frame_geometry.width() - self.text_cell - 2) / col_num
        cell_height = (frame_geometry.height() - self.text_cell - 2) / row_num

        self.cell_width = min(cell_width, cell_height)

    @staticmethod
    def clear_table(table):
        """
        Clear table view by set all cell to new item
        """
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                item = QTableWidgetItem()
                table.setItem(row, col, item)

    def update_table(self):
        """
        Update alignment table widget
        """
        if not self.align_mats or self.align_id < 0:
            self.statusBar.showMessage('Warning: no alignment loaded')
            return

        str_a = self.align_mats[self.align_id].nlg[0][0]
        str_b = self.align_mats[self.align_id].nlg[0][1]
        str_c = self.align_mats[self.align_id].nlg[1][0]
        str_d = self.align_mats[self.align_id].nlg[1][1]

        ab_mat = self.align_mats[self.align_id].mats[0]
        ac_mat = self.align_mats[self.align_id].mats[1]
        bd_mat = self.align_mats[self.align_id].mats[2]
        cd_mat = self.align_mats[self.align_id].mats[3]

        len_d = len(str_b) + len(str_c) - len(str_a)

        col_num = len(str_b) + len(str_c) + 1
        row_num = len(str_a) + len_d + 1

        # Set cell size
        self.tableAlignment.setColumnCount(col_num)
        self.tableAlignment.setRowCount(row_num)

        for col_id in range(col_num):
            self.tableAlignment.setColumnWidth(col_id, self.cell_width)

        for row_id in range(row_num):
            self.tableAlignment.setRowHeight(row_id, self.cell_width)

        # Text cells
        self.tableAlignment.setColumnWidth(len(str_b), self.text_cell)
        self.tableAlignment.setRowHeight(len(str_a), self.text_cell)

        self.clear_table(self.tableAlignment)

        # Cell color
        # AB
        for cell_id in itertools.product(range(len(str_a)), range(len(str_b))):
            cell_value = 255 * (1 - ab_mat[cell_id])
            item = QTableWidgetItem()
            item.setBackground(QtGui.QColor(cell_value, cell_value, cell_value))
            if self.actionEqualPoints.isChecked() and str_a[len(str_a) - cell_id[0] - 1] == \
                    str_b[len(str_b) - cell_id[1] - 1]:
                item.setText(self.equal_char)
                item.setForeground(QtGui.QColor(255, 0, 0))
                item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.tableAlignment.setItem(cell_id[0], cell_id[1], item)

        # AC
        for cell_id in itertools.product(range(len(str_a)), range(len(str_c))):
            cell_value = 255 * (1 - ac_mat[cell_id])
            item = QTableWidgetItem()
            item.setBackground(QtGui.QColor(cell_value, cell_value, cell_value))
            if self.actionEqualPoints.isChecked() and str_a[len(str_a) - cell_id[0] - 1] == str_c[cell_id[1]]:
                item.setText(self.equal_char)
                item.setForeground(QtGui.QColor(255, 0, 0))
                item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.tableAlignment.setItem(cell_id[0], len(str_b) + cell_id[1] + 1, item)

        # BD
        for cell_id in itertools.product(range(len(str_d)), range(len(str_b))):
            cell_value = 255 * (1 - bd_mat[cell_id])
            item = QTableWidgetItem()
            item.setBackground(QtGui.QColor(cell_value, cell_value, cell_value))
            if self.actionEqualPoints.isChecked() and str_d[cell_id[0]] == str_b[len(str_b) - cell_id[1] - 1]:
                item.setText(self.equal_char)
                item.setForeground(QtGui.QColor(255, 0, 0))
                item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.tableAlignment.setItem(len(str_a) + cell_id[0] + 1, cell_id[1], item)

        # CD
        for cell_id in itertools.product(range(len(str_d)), range(len(str_c))):
            cell_value = 255 * (1 - cd_mat[cell_id])
            item = QTableWidgetItem()
            item.setBackground(QtGui.QColor(cell_value, cell_value, cell_value))
            if self.actionEqualPoints.isChecked() and str_d[cell_id[0]] == str_c[cell_id[1]]:
                item.setText(self.equal_char)
                item.setForeground(QtGui.QColor(255, 0, 0))
                item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.tableAlignment.setItem(len(str_a) + cell_id[0] + 1, len(str_b) + cell_id[1] + 1, item)

        # Axes text
        # A
        for word_id, word in enumerate(str_a):
            item = QTableWidgetItem(word)
            item.setBackground(QtGui.QColor(232, 98, 123))  # momiji
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.tableAlignment.setItem(len(str_a) - word_id - 1, len(str_b), item)

        # B (Vertical)
        for word_id, word in enumerate(str_b):
            item = QTableWidgetItem('\n'.join(list(word)))
            item.setBackground(QtGui.QColor(238, 138, 76))  # yu-yake
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.tableAlignment.setItem(len(str_a), len(str_b) - word_id - 1, item)

        # C (Vertical)
        for word_id, word in enumerate(str_c):
            item = QTableWidgetItem('\n'.join(list(word)))
            item.setBackground(QtGui.QColor(83, 131, 151))  # tsuki-yo
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.tableAlignment.setItem(len(str_a), len(str_b) + word_id + 1, item)

        # D
        for word_id in range(len_d):
            if word_id < len(str_d):
                item = QTableWidgetItem(str_d[word_id])
            else:
                item = QTableWidgetItem()
            item.setBackground(QtGui.QColor(182, 210, 110))  # chiku-rin
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.tableAlignment.setItem(len(str_a) + word_id + 1, len(str_b), item)

        # Dead center
        item = QTableWidgetItem()
        self.tableAlignment.setItem(len(str_a), len(str_b), item)
        self.update_status('Table updated. Alignment #%d' % (self.align_id + 1))

    def update_status(self, message):
        self.statusBar.showMessage(message)


###############################################################################

def read_argv():
    this_program = 'v%s (c) %s %s' % (__version__, __date__.split('/')[2], __author__)
    this_description = __description__

    parser = argparse.ArgumentParser(prog=this_program, description=this_description)
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='verbose mode')

    return parser.parse_args()


def read_input(input_path):
    """
    Read input from train file and experiments file
    :type input_path: path to file contains list of question in each line
    :return: list of sentence (pair of input and output)
    """

    # Read input file
    alg_mats = []

    if '.gz' in input_path:
        alg_file = gzip.open(input_path, 'rt', encoding='utf8')
    else:
        alg_file = open(input_path, 'r', encoding='utf8')

    for alg in alg_file.readlines():
        alg_mats.append(AnalogyAlignment(alg.rstrip('\n')))
    alg_file.close()

    return alg_mats


def main():
    viz = VisualizerInterface()
    viz.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('interface/app.png'))
    args = read_argv()
    main()
