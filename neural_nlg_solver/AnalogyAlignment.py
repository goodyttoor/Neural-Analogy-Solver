#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import sys

import numpy as np

from neural_nlg_solver.AnalogyEquation import AnalogyEquation

###############################################################################

__author__ = 'KAVEETA Vivatchai <vivatchai@fuji.waseda.jp, goodytong@gmail.com>'

# __date__, __version__ = '17/01/2017', '0.1'  # First version
# __date__, __version__ = '10/02/2017', '0.2'  # AlignmentMatrix -> AnalogyAlignment
__date__, __version__ = '23/03/2017', '0.3'  # Mat. dim. now determined by string itself

__description__ = 'Class for handling alignment matrix'


###############################################################################

class AnalogyAlignment:
    """
    Class for handling alignment matrix
    """

    nlg = None
    alg_mats = [None, None, None, None]

    def __init__(self, *args):
        """
        Initialize alignment matrix
        """

        # String input in following format
        # (A : B :: C : D   [0,1,..] [0,1,..] [0,1,..] [0,1,..])
        if len(args) == 1:

            if type(args[0]) == AnalogyEquation:
                self.nlg = AnalogyEquation(args[0])

                # Set all alignment matrices to None
                self.alg_mats = [None, None, None, None]

            else:
                # Split string into 2 parts
                parts = args[0].split('\t')

                # AnalogyEquation file
                if len(parts) == 1:

                    # First part is string equation
                    self.nlg = AnalogyEquation(parts[0])

                    # Set all alignment matrices to None
                    self.alg_mats = [None, None, None, None]

                # AnalogyAlignment file
                elif len(parts) == 2:

                    # First part is string equation
                    self.nlg = AnalogyEquation(parts[0])

                    # Second part is alignment matrices
                    mat_strs = parts[1].split(' ')

                    # Temporary set default values
                    self.alg_mats = [None, None, None, None]

                    # Convert literal strings back to numpy matrices
                    for mat_id, mat_str in enumerate(mat_strs):
                        self.alg_mats[mat_id] = self.str_to_mat(mat_str)

    def __repr__(self):
        """
        String representation of this object
        :return: string representation
                 A : B :: C : D   [0,1,...] [0,1,...] [0,1,...] [0,1,...]
        """

        return '{0}\t{1} {2} {3} {4}'.format(self.nlg, self.mat_to_str(self.mats[0]), self.mat_to_str(self.mats[1]),
                                             self.mat_to_str(self.mats[2]), self.mat_to_str(self.mats[3]))

    def __hash__(self):
        return hash(str(self))

    @property
    def strs(self):
        """
        Get strings in analogy equation
        :return: list of string (list of symbols)
        """

        return self.nlg[0][0], self.nlg[0][1], self.nlg[1][0], self.nlg[1][1]

    @property
    def mats(self):
        """
        Get list of alignment matrices
        :return: list of numpy arrays
        """

        # If string analogical equation, matrices variable is None
        # Build alignment matrices on the fly

        # Matrix AB
        if self.alg_mats[0] is not None:
            mat_ab = self.alg_mats[0]
        else:
            mat_ab = np.flipud(np.fliplr(self.gen_align_matrix(self.nlg[0][0], self.nlg[0][1])))

        # Matrix AC
        if self.alg_mats[1] is not None:
            mat_ac = self.alg_mats[1]
        else:
            mat_ac = np.flipud(self.gen_align_matrix(self.nlg[0][0], self.nlg[1][0]))

        # Matrix DB
        if self.alg_mats[2] is not None:
            mat_db = self.alg_mats[2]
        else:
            mat_db = np.fliplr(self.gen_align_matrix(self.nlg[1][1], self.nlg[0][1]))

        # Matrix DC
        if self.alg_mats[3] is not None:
            mat_dc = self.alg_mats[3]
        else:
            mat_dc = self.gen_align_matrix(self.nlg[1][1], self.nlg[1][0])

        return [mat_ab, mat_ac, mat_db, mat_dc]

    @property
    def tex_str(self):
        """
        Get latex code to display current alignment
        """

        # Get normal string
        str_a = ''.join(self.nlg[0][0])
        str_b = ''.join(self.nlg[0][1])
        str_c = ''.join(self.nlg[1][0])
        str_d = ''.join(self.nlg[1][1])

        # Not soft alignment, use simple alignment command (aligmat.sty)
        if self.alg_mats == [None, None, None, None]:

            # Generate latex code
            tex_str = '\\alignmat[0.5cm]{{{0}}}{{{1}}}{{{2}}}{{{3}}}'.format(str_a, str_b, str_c, str_d)

        # Soft alignment, use advance alignment command (alignmatadv.sty)
        else:

            # Generate latex code
            # Strings part
            tex_str = '\\alignmatadv[0.5cm]{{{0}}}{{{1}}}{{{2}}}{{{3}}}'.format(str_a, str_b, str_c, str_d)

            # Matrices part
            tex_str += '{{{0}}}{{{1}}}{{{2}}}{{{3}}}'.format(self.mat_to_tex_str(self.mats[0]),
                                                             self.mat_to_tex_str(self.mats[1]),
                                                             self.mat_to_tex_str(self.mats[2]),
                                                             self.mat_to_tex_str(self.mats[3]))

        return tex_str

    @staticmethod
    def gen_align_matrix(string_a, string_b):
        """
        Generate the alignment matrix from a pair of input string
        :param string_a: source string (axis:0 in output)
        :param string_b: target string (axis:1 in output)
        :return: matrix with the size of [mat_len x mat_len]
        """

        mat = np.zeros((len(string_a), len(string_b)), dtype=float)

        for idx_a, char_a in enumerate(string_a):
            for idx_b, char_b in enumerate(string_b):
                if char_a == char_b:
                    mat[idx_a, idx_b] = 1
        return mat

    @staticmethod
    def mat_to_str(np_mat):
        """
        Convert numpy matrix to string
        :param np_mat: Numpy matrix or array
        :return: string representation
        """

        np.set_printoptions(threshold=np.nan)
        return np.array2string(np_mat, precision=2, suppress_small=True, separator=',', max_line_width=sys.maxsize) \
            .replace(' ', '').replace('\n', '').replace('.,', ',').replace('.]', ']')

    @staticmethod
    def mat_to_tex_str(np_mat):
        """
        Convert matrix to latex array string literal
        :param np_mat: Numpy matrix or array
        :return: latex string
        """

        # Normalize to range [0,100]
        tmp_mat = np_mat.copy()
        tmp_mat *= 100

        np.set_printoptions(threshold=np.nan)
        return np.array2string(tmp_mat.astype(int), precision=2, suppress_small=True, separator=',',
                               max_line_width=sys.maxsize) \
            .replace('[', '').replace(' ', '').replace('\n', '').replace('.,', ',').replace('.]', ']') \
            .replace('],', '\\\\').replace(']', '')

    @staticmethod
    def str_to_mat(mat_str):
        """
        Convert numpy matrix to string
        :param mat_str: Input string in [0,1,...] format
        :return: numpy array object
        """

        # Use json package for super easy parsing
        return np.array(json.loads(mat_str))
