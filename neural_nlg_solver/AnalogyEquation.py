#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
from itertools import product

from neural_nlg_solver.AnalogyRatio import AnalogyRatio

###############################################################################

__author__ = 'KAVEETA Vivatchai <vivatchai@fuji.waseda.jp, goodytong@gmail.com>'

__date__, __version__ = '17/01/2017', '0.1'  # First version

__description__ = 'Class for handling analogical equation'


###############################################################################

class AnalogyEquation:
    """
    Class for handling analogical equation
    """

    def __init__(self, *args):
        """
        Initialize equation
        """

        # Handle different types of input
        if len(args) == 1:
            if type(args[0]) == AnalogyEquation:
                self.ratios = args[0].ratios
            else:
                ratios = re.split('[ ]*::[ ]*', args[0])
                if len(ratios) == 2:
                    self.ratios = [AnalogyRatio(ratios[0]), AnalogyRatio(ratios[1])]
        elif len(args) == 2:
            if type(args[0]) == AnalogyRatio:
                self.ratios = [args[0], args[1]]
            else:
                self.ratios = [AnalogyRatio(args[0]), AnalogyRatio(args[1])]
        elif len(args) == 4:
            self.ratios = [AnalogyRatio(args[0], args[1]),
                           AnalogyRatio(args[2], args[3])]
        else:
            self.ratios = [AnalogyRatio(), AnalogyRatio()]

    def __repr__(self):
        """
        Standard print string format
        :return: string in (a : b : c : d) format
        """
        return '{} :: {}'.format(*self.ratios)

    def __hash__(self):
        return hash(str(self))

    def __getitem__(self, index):
        """
        Get list of ratio in equation
        :param index: position of ratio [0,1], 
                      to get a string inside ratio uses: eq[0][0] = A
        :return: ratio at position index-th
        """
        return self.ratios[index]

    def __setitem__(self, index, ratio):
        """
        Set string in equation
        :param index: position of string [0,3]
        :param ratio: ratio to set
        """
        if type(ratio) == AnalogyRatio:
            self.ratios[index] = ratio
        else:
            self.ratios[index] = AnalogyRatio(ratio)

    def __eq__(self, other):
        """
        Equality of analogical equation
        :param other: other AnalogyEquation to be compared with
        :return: {type: boolean} True: equal, False: not equal
        """
        if str(self) == str(other):
            return True
        return False

    @property
    def strs(self):
        """
        Return list of string
        :return: list of 4 string
        """
        return [self.ratios[ratio_id][str_idx] for ratio_id, str_idx in product([0, 1], [0, 1])]
