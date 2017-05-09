#!/usr/bin/python
# -*- coding: utf-8 -*-

import re

###############################################################################

__author__ = 'KAVEETA Vivatchai <vivatchai@fuji.waseda.jp, goodytong@gmail.com>'

__date__, __version__ = '17/01/2017', '0.1'  # First version

__description__ = 'Class for handling analogical ratio'


###############################################################################

class AnalogyRatio:
    """
    Class for handling analogical ratio
    """

    def __init__(self, *args):
        """
        Initialize equation
        """

        # Handle different types of input
        if len(args) == 1:

            strings = re.split('[ ]+:[ ]+', args[0])
            if len(strings) == 2:

                # If bucket found, Try parsing (list literal)
                if strings[0][0] == '[' \
                        and strings[0][-1] == ']' \
                        and strings[1][0] == '[' \
                        and strings[1][-1] == ']':

                    self.nlg_strs = [strings[0][1:-1].split(','),
                                     strings[1][1:-1].split(',')]

                # Fallback to splitting string to characters
                else:
                    self.nlg_strs = [list(strings[0]),
                                     list(strings[1])]

        elif len(args) == 2:

            if type(args[0]) == list and type(args[1]) == list:
                self.nlg_strs = [args[0], args[1]]

            else:
                if args[0][0] == '[' and args[0][-1] == ']' \
                        and args[1][0] == '[' and args[1][-1] == ']':

                    self.nlg_strs = [args[0][1:-1].split(','),
                                     args[1][1:-1].split(',')]

                # Fallback to splitting string to characters
                else:
                    self.nlg_strs = [list(args[0]), list(args[1])]

        else:
            self.nlg_strs = [[], []]

    def __repr__(self):
        """
        Standard print string format
        :return: string in (a : b) format
        """
        return '{} : {}'.format(*['[{}]'.format(','.join(string))
                                  for string in self.nlg_strs])

    def __hash__(self):
        return hash(str(self))

    def __getitem__(self, index):
        """
        Get string in equation
        :param index: position of string [0,1]
        :return: string at position index-th
        """
        return self.nlg_strs[index]

    def __setitem__(self, index, string):
        """
        Set string in equation
        :param index: position of string [0,1]
        :param string: string to set
        """
        self.nlg_strs[index] = string

    def __eq__(self, other):
        """
        Equality of analogical ratio
        :param other: other AnalogyRatio to be compared with
        :return: {type: boolean} True: equal, False: not equal
        """
        if str(self) == str(other):
            return True

        return False
