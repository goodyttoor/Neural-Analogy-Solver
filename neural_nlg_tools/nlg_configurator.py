#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QFileDialog, \
    QGraphicsPixmapItem, QGraphicsScene, QMessageBox
from PyQt5.uic import loadUiType
from keras.utils.vis_utils import plot_model

from neural_nlg_solver.AnalogyNeuralModel import FullyConnectedModel

# Hide tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Ui_MainWindow, QMainWindow = loadUiType('interface/nlg_configurator.ui')

###############################################################################

__author__ = 'KAVEETA Vivatchai <vivatchai@fuji.waseda.jp, goodytong@gmail.com>'

# __date__, __version__ = '17/01/2017', '0.1'  # Prototype
# __date__, __version__ = '06/02/2017', '0.2'  # Change some default parameters
# __date__, __version__ = '07/02/2017', '0.3'  # Dense only, finalize structure
# __date__, __version__ = '23/03/2017', '0.4'  # Non-uniform model
# __date__, __version__ = '28/03/2017', '0.5'  # Remove length -> on-the-fly training models
__date__, __version__ = '07/04/2017', '0.6'  # Remove node num -> using average input, output mat. dim.

__description__ = 'Configure neural network model for solving formal analogy equations'


###############################################################################

# Lists of available layer parameters
optimizers = ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
losses = ['Binary_Crossentropy', 'Mean_Squared_Error', 'Mean_Absolute_Error', 'Mean_Squared_Logarithmic_Error',
          'Squared_Hinge', 'Hinge', 'Categorical_Crossentropy', 'Sparse_Categorical_Crossentropy',
          'Kullback_Leibler_Divergence', 'Poisson', 'Cosine']
activations = ['Sigmoid', 'Softmax', 'Softplus', 'Softsign', 'ReLU', 'Tanh', 'Hard_Sigmoid', 'Linear']


###############################################################################

def align_combo_box(combo_box, align):
    """
    Align all components in combobox
    :param combo_box: the parent combobox
    :param align: alignment to set {type: QtCore.Qt.Alignment}
    """

    combo_box.setEditable(True)
    combo_box.lineEdit().setAlignment(align)
    combo_box.lineEdit().setReadOnly(True)

    # Get all children of combobox
    for i in range(combo_box.count()):
        combo_box.setItemData(i, align, Qt.TextAlignmentRole)


###############################################################################

class ConfiguratorInterface(QMainWindow, Ui_MainWindow):
    # Neural network layers
    layers = []

    def __init__(self):
        """
        Initialize interface components
        """

        # Setup interface
        super(ConfiguratorInterface, self).__init__()

        self.setupUi(self)
        self.center_window()

        # Set action handlers (user click)
        self.actionNew.triggered.connect(self.action_new)
        self.actionOpen.triggered.connect(self.action_load_config)
        self.actionSaveConfig.triggered.connect(self.action_save_config)
        self.actionExit.triggered.connect(self.close)
        self.actionAbout.triggered.connect(self.action_about)
        self.actionPreview.triggered.connect(self.action_preview)
        self.spinBox_compute_num.valueChanged.connect(self.action_layer_change)

        # Initialize default setting
        self.action_layer_change()
        self.action_new()
        self.layers = []
        self.update_status('Ready to rock the world')

    @staticmethod
    def closeEvent(event):
        # Clean temp preview file
        if os.path.exists('preview.png'):
            os.remove('preview.png')
        event.accept()

    def update_status(self, message):
        """
        Update text on bottom status bar
        :param message: string to be displayed
        """
        self.statusBar.showMessage(message)

    def center_window(self):
        """
        Move windows to center of the current monitor
        """
        # Get frame object
        qr = self.frameGeometry()

        # Move to center
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def action_new(self):
        """
        Handler when user click new in file menu
        """
        # Reset default values to all parameters
        self.comboBox_basic_optimizer.clear()
        self.comboBox_basic_optimizer.addItems(optimizers)
        align_combo_box(self.comboBox_basic_optimizer, Qt.AlignCenter)
        self.comboBox_basic_optimizer.setCurrentIndex(0)

        self.comboBox_basic_loss.clear()
        self.comboBox_basic_loss.addItems(losses)
        align_combo_box(self.comboBox_basic_loss, Qt.AlignCenter)
        self.comboBox_basic_loss.setCurrentIndex(0)

        self.comboBox_basic_activation.clear()
        self.comboBox_basic_activation.addItems(activations)
        align_combo_box(self.comboBox_basic_activation, Qt.AlignCenter)
        self.comboBox_basic_activation.setCurrentIndex(0)

        self.spinBox_basic_epochs.setValue(200)
        self.spinBox_basic_batch.setValue(64)
        self.spinBox_compute_num.setValue(0)

        # Update 'new' state
        self.update_status('New file')

    @staticmethod
    def action_about():
        """
        Handler when user click about in help menu
        Display message box about me :)
        """
        msg = QMessageBox()
        msg.setWindowTitle('About')
        msg.setText('By KAVEETA Vivatchai')
        msg.setInformativeText('EBMT/NLP Lab, 2016-2017')
        msg.setIconPixmap(QPixmap('interface/avatar.png'))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def action_layer_change(self):
        """
        Handler when user change layer number
        """
        # Create new layer widget
        while self.spinBox_compute_num.value() > len(self.layers):
            new_item = DenseLayer(len(self.layers), self.groupBox_compute)

            # If not the first layer, copy setting currently last layer
            if len(self.layers) > 0:
                new_item.activation = self.layers[-1].activation
                new_item.dropout = self.layers[-1].dropout

            # Add new widget to bottom of list
            self.verticalLayout_compute.addWidget(new_item)
            self.layers.append(new_item)

        # If number reduced, remove tail item(s)
        while self.spinBox_compute_num.value() < len(self.layers):
            self.layers.pop().setParent(None)

        # Update 'layer changed' state
        self.update_status('Number of layer changed')

    def action_preview(self):
        """
        Handler when user click build button in toolbar
        """
        params = self.get_params()

        dense_model = FullyConnectedModel()
        dense_model.config([1, 2, 3, 4], params['layers'], params['optimizer'], params['loss'], params['activation'])
        plot_model(dense_model.model, to_file='preview.png')

        # Create new scene
        scene = QGraphicsScene()
        pix_map = QPixmap("preview.png")
        scene.addItem(QGraphicsPixmapItem(pix_map))

        # Set to graphic view
        self.graphicsView.setScene(scene)
        self.update_status('Preview updated')

    def get_params(self):
        """
        Get values of parameters from interface components
        """
        # Get layer(s) parameters
        layers_params = []
        for layer in self.layers:
            config = {'activation': layer.activation, 'dropout': round(layer.dropout, 2)}
            layers_params.append(config)

        # Return all parameters
        return {'optimizer': self.comboBox_basic_optimizer.currentText(),
                'loss': self.comboBox_basic_loss.currentText(),
                'activation': self.comboBox_basic_activation.currentText(),
                'epochs': self.spinBox_basic_epochs.value(),
                'batch': self.spinBox_basic_batch.value(),
                'layers': layers_params}

    def set_params(self, params):
        """
        Set value of basic parameters to interface components
        """
        # Reset to default value
        self.action_new()

        # Set parameters from config file back to interface components
        self.comboBox_basic_optimizer.setCurrentText(params['optimizer'])
        self.comboBox_basic_loss.setCurrentText(params['loss'])
        self.comboBox_basic_activation.setCurrentText(params['activation'])
        self.spinBox_basic_epochs.setValue(params['epochs'])
        self.spinBox_basic_batch.setValue(params['batch'])

        # Layer(s) parameters
        self.spinBox_compute_num.setValue(len(params['layers']))
        for idx, config in enumerate(params['layers']):
            self.layers[idx].activation = config['activation']
            self.layers[idx].dropout = config['dropout']

    def action_load_config(self):
        """
        Handler when user click load in file menu
        """
        # Open load file dialog
        in_path = QFileDialog.getOpenFileName(None, 'Open Config File', filter='config (*.cfg)')

        # If user cancel, do not load
        if in_path[0] == '':
            return

        # Update internal parameters
        with open(in_path[0], 'r', encoding='utf8') as in_file:
            self.set_params(json.load(in_file))

        # Update 'loaded' state
        self.update_status('Config loaded')

    def action_save_config(self):
        """
        Handler when user click save in file menu
        """
        # Display save file dialog
        out_path = QFileDialog.getSaveFileName(None, 'Save Config File', filter='config (*.cfg)')[0]

        # If user cancel, do not save
        if out_path == '':
            return

        # Append .cfg type if not already existed
        if '.cfg' not in out_path:
            out_path += '.cfg'

        # Save config file
        with open(out_path, 'w', encoding='utf8') as out_file:
            json.dump(self.get_params(), out_file)

        # Update 'saved' state
        self.update_status('Config saved')


###############################################################################

class DenseLayer(QtWidgets.QGroupBox):
    """
    Class represented fully connected layer interface widget group
    These will be appended to the parent widget at some point
    """

    id = None
    comboBox_activation = None
    doubleSpinBox_dropout = None

    def __init__(self, compute_id, parent=None):
        # New fully connected (dense) layer widget
        self.id = compute_id
        QtWidgets.QGroupBox.__init__(self, parent)

        # Set sizes and positions
        self.setTitle("Layer " + str(self.id + 1))
        form_layout = QtWidgets.QFormLayout(self)
        form_layout.setContentsMargins(11, 11, 11, 11)
        form_layout.setSpacing(6)
        label_size = QtWidgets.QLabel(self)
        form_layout.setWidget(0, QtWidgets.QFormLayout.LabelRole, label_size)

        # 1.Labels (left side)
        label_activation = QtWidgets.QLabel(self)
        label_activation.setText("Activation")
        form_layout.setWidget(1, QtWidgets.QFormLayout.LabelRole, label_activation)
        label_dropout = QtWidgets.QLabel(self)
        label_dropout.setText("Dropout")
        form_layout.setWidget(2, QtWidgets.QFormLayout.LabelRole, label_dropout)

        # 2.Form objects (right side)
        # 2.1 Activation combobox
        self.comboBox_activation = QtWidgets.QComboBox(self)
        self.comboBox_activation.addItems(activations)
        form_layout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_activation)
        align_combo_box(self.comboBox_activation, Qt.AlignCenter)

        # 2.2 Dropout value spinbox
        self.doubleSpinBox_dropout = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox_dropout.setAlignment(QtCore.Qt.AlignCenter)
        self.doubleSpinBox_dropout.setDecimals(1)
        self.doubleSpinBox_dropout.setSingleStep(0.1)
        form_layout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_dropout)

        # Set default values
        self.reset()

    def reset(self):
        """
        Reset parameters in layer to default values
        :return:
        """
        self.comboBox_activation.setCurrentIndex(0)
        self.doubleSpinBox_dropout.setValue(0.0)

    @property
    def activation(self):
        """
        Get activation function of this layer
        :return: activation function name {type: string}
        """
        return self.comboBox_activation.currentText()

    @activation.setter
    def activation(self, value):
        """
        Set activation function
        :param value: activation function name
        """
        self.comboBox_activation.setCurrentText(value)

    @property
    def dropout(self):
        """
        Get dropout value of this layer
        :return: dropout value {type: float}
        """
        return self.doubleSpinBox_dropout.value()

    @dropout.setter
    def dropout(self, value):
        """
        Set dropout value of this layer
        :param value: dropout value
        """
        self.doubleSpinBox_dropout.setValue(value)


###############################################################################

def main():
    # Show UI
    viz = ConfiguratorInterface()
    viz.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('interface/app.png'))
    main()
