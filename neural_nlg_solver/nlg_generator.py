#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import gzip
import math
import multiprocessing
import sys
import time
from collections import defaultdict
from itertools import product, repeat

import matplotlib.pyplot as plt

from neural_nlg_solver import AnalogyAlignment
from neural_nlg_solver.AnalogyEquation import AnalogyEquation

plt.style.use('ggplot')

###############################################################################

__author__ = 'KAVEETA Vivatchai <vivatchai@fuji.waseda.jp, goodytong@gmail.com>'

# __date__, __version__ = '10/01/2017', '0.1'  # First version
# __date__, __version__ = '11/01/2017', '0.2'  # Fix progressbar
# __date__, __version__ = '12/01/2017', '0.3'  # Multiprocessing
# __date__, __version__ = '13/01/2017', '0.4'  # Fix combination algorithm, Doctest
# __date__, __version__ = '14/01/2017', '0.5'  # File writer thread, filtering
# __date__, __version__ = '15/01/2017', '0.6'  # Cluster limit
# __date__, __version__ = '16/01/2017', '0.7'  # Fix leaky thread bug,
#                                              # Option to read clusters file
#                                              # New doctests
# __date__, __version__ = '17/01/2017', '0.8'  # Fix doctest
# __date__, __version__ = '19/01/2017', '0.9'  # AnalogyRatio and AnalogyCluster classes
# __date__, __version__ = '27/01/2017', '1.0'  # Fix generation bugs
# __date__, __version__ = '10/02/2017', '1.1'  # On the fly gzipping, fix doctests
# __date__, __version__ = '11/02/2017', '2.0'  # Version 2: major overhaul.
#                                              # Formalize and separate generation functions
#                                              # Remove clustering. Add inserting
# __date__, __version__ = '15/02/2017', '2.1'  # Proper inserting and casting functions
#                                              # Seperate base analogy loops to reduce memory usage
# __date__, __version__ = '17/02/2017', '2.2'  # Memory optimization for fix_len arguments
# __date__, __version__ = '18/02/2017', '2.3'  # Implement new multi-processing
# __date__, __version__ = '10/03/2017', '2.4'  # Non-uniform fixed length and variable length
# __date__, __version__ = '28/03/2017', '2.5'  # Fix inserting bug
# __date__, __version__ = '05/04/2017', '2.6'  # Single thread fallback. Temporary use single thread as default
# __date__, __version__ = '06/04/2017', '2.7'  # Filter only unique analogies
# __date__, __version__ = '16/04/2017', '2.8'  # Fix memory for multi-thread mode, finally
#                                              # Default to multi thread mode
__date__, __version__ = '05/05/2017', '2.9'  # Fix severe bug in casting (suffix in C). Redo all experiments ;o;

__description__ = 'Generate analogy equations from list of base analogies'


# IDEA: Hashing filter?

# IDEA: Each loop generate by every methods
#     : Save the analogy which have the target length to output list
#     : Other compare to before list remove any duplication (after set - before set)
#     : Do until all the list is empty

###############################################################################

def read_argv():
    program = 'v%s (c) %s %s' % (__version__, __date__.split('/')[2], __author__)
    description = __description__

    parser = argparse.ArgumentParser(prog=program, description=description)

    # Required arguments
    parser.add_argument('file', action='store', help='an input file contains list of analogies')
    parser.add_argument('length', action='store', type=int, default=[8], help='string lengths (1 or 4 items)',
                        nargs='*')

    # Generating arguments
    parser.add_argument('-vl', '--var-length', action='store_true', dest='var_length', default=False,
                        help='give all variable length analogies as results')
    parser.add_argument('-m', '--methods', action='store', dest='methods', default='pmric',
                        help='methods to run (default: pmric), r = reduplicating, p = permuting, m = mirroring, '
                             'i = inserting, c = casting')

    # Output arguments
    parser.add_argument('-o', '--out', action='store', dest='output', default='sys.stdout', help='output path')

    # Miscellaneous arguments
    parser.add_argument('-j', '--threads', action='store', type=int, dest='threads',
                        default=multiprocessing.cpu_count(), help='number of processing threads')
    parser.add_argument('-g', '--graph', action='store_true', dest='graph', default=False,
                        help='display statistical chart after generation')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='verbose mode')

    tmp_args = parser.parse_args()

    if len(tmp_args.length) != 1 and len(tmp_args.length) != 4:
        print('Error: length must be either 1 or 4 items(s)', file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    return tmp_args


###############################################################################

def read_nlg_file(in_path):
    """
    Read in analogies from input file.
    Assuming standard format as each line contains one analogical equation separated by ' : ' and ' :: '
    ex.- a : b :: c : d
    :param in_path: input file path contains list of analogical equations, one per each line
    """

    nlgs = []

    # Start timer
    start_time = time.time()

    if args.verbose:
        print('\nRead analogies from: {}'.format(in_path), file=sys.stderr)

    # Open file to read
    if '.gz' in in_path:
        in_file = gzip.open(in_path, 'rt', encoding='utf8')
    else:
        in_file = open(in_path, 'r', encoding='utf8')

    # with open(in_path) as in_file:
    for line in in_file.readlines():

        # Create new analogy object
        new_eq = AnalogyEquation(line.rstrip())

        # Pre-filter input equations. Ignore invalid equation
        if new_eq:
            nlgs.append(new_eq)

    # Close file
    in_file.close()

    if args.verbose:
        exec_time = time.time() - start_time
        print('   - Finished in: {0:.2f} sec'.format(exec_time), file=sys.stderr)
        print('   - Number of analogies: {}'.format(len(nlgs)), file=sys.stderr)

    # Display string length chart
    if args.graph:

        # Loop all analogy
        str_len_count = defaultdict(int)
        for nlg in nlgs:
            # Average length of string in the equation
            avg_len = math.floor(sum(len(string) for ratios in nlg for string in ratios) / 4)
            str_len_count[avg_len] += 1

        x_data = list(str_len_count.keys())
        y_data = list(str_len_count.values())

        plt.title('Number of equations by average string length')
        plt.xlabel('Average length of string')
        plt.ylabel('Number of analogical equations')

        plt.bar(x_data, y_data)
        plt.show()

    return nlgs


###############################################################################

def is_fix_len(nlg, str_len):
    """
    Check if AnalogicalEquation is fix_len
    :param nlg: AnalogicalEquation object
    :param str_len: String lengths
    """

    return len(nlg[0][0]) == str_len[0] and len(nlg[0][1]) == str_len[1] and \
        len(nlg[1][0]) == str_len[2] and len(nlg[1][1]) == str_len[3]


def is_within_len(nlg, str_len):
    """
    Check if strings in AnalogicalEquation are shorter or equal to str_len
    :param nlg: AnalogicalEquation object
    :param str_len: String lengths
    """

    return len(nlg[0][0]) <= str_len[0] and len(nlg[0][1]) <= str_len[1] and \
        len(nlg[1][0]) <= str_len[2] and len(nlg[1][1]) <= str_len[3]


###############################################################################


def gen_permute(nlg, str_len, fix_len=False):
    """
    Generate all permuted forms of the list of input equations
    :param nlg: an AnalogyEquation object
    :param str_len: length of all strings in generated equations
    :param fix_len: get only fixed length equations
    """

    # All equivalent combinations (8)
    eq_forms = [[0, 1, 2, 3], [0, 2, 1, 3], [1, 0, 3, 2], [1, 3, 0, 2],
                [2, 0, 3, 1], [2, 3, 0, 1], [3, 1, 2, 0], [3, 2, 1, 0]]

    # List of generated equations
    gen_nlgs = []

    # Generate all form
    for form in eq_forms:

        eq_nlg = AnalogyEquation(*[nlg[int(math.floor(idx / 2))][idx % 2] for idx in form])

        # Add to generated list
        if is_within_len(eq_nlg, str_len) and (not fix_len or is_fix_len(eq_nlg, str_len)):
            gen_nlgs.append(eq_nlg)

    return gen_nlgs


def gen_mirror(nlg, str_len, fix_len=False):
    """
    Mirror analogy equation
    :param nlg: an AnalogyEquation object
    :param str_len: length of all strings in generated equations
    :param fix_len: get only fixed length equations
    """

    # List of generated equations
    gen_nlgs = []

    # Append normal form
    if is_within_len(nlg, str_len) and (not fix_len or is_fix_len(nlg, str_len)):
        gen_nlgs.append(nlg)

    mirrored_nlg = AnalogyEquation()
    for ratio_idx, str_idx in product([0, 1], [0, 1]):
        mirrored_nlg[ratio_idx][str_idx] = nlg[ratio_idx][str_idx][::-1]

    # Add to generated list
    if is_within_len(mirrored_nlg, str_len) and (not fix_len or is_fix_len(mirrored_nlg, str_len)):
        gen_nlgs.append(mirrored_nlg)

    return gen_nlgs


def gen_redup(nlg, str_len, fix_len=False):
    """
    Reduplicate equations to maximum length
    :param nlg: an AnalogyEquation object
    :param str_len: length of all strings in generated equations
    :param fix_len: get only fixed length equations
    """

    # List of generated equations
    gen_nlgs = []

    # Calculate maximum reduplication number
    max_redup = int(math.floor(min(str_len[0] / len(nlg[0][0]), str_len[1] / len(nlg[0][1]),
                                   str_len[2] / len(nlg[1][0]), str_len[3] / len(nlg[1][1]))))

    for redup_id in range(1, max_redup + 1):

        # Create a new AnalogyEquation
        duped_nlg = AnalogyEquation()

        for ratio_idx, str_idx in product([0, 1], [0, 1]):

            # Build new list of symbols = new string
            new_string = []
            for symbol in nlg[ratio_idx][str_idx]:
                new_string.extend([symbol] * redup_id)

            # Set new string to new AnalogyEquation
            duped_nlg[ratio_idx][str_idx] = new_string

        # Add to generated list
        if not fix_len or is_fix_len(duped_nlg, str_len):
            gen_nlgs.append(duped_nlg)

    return gen_nlgs


def gen_insert(nlg, str_len, fix_len=False):
    """
    Inserting equations to maximum length
    :param nlg: an AnalogyEquation object
    :param str_len: length of all strings in generated equations
    :param fix_len: get only fixed length equations
    :return: list of inserted equations
    """

    # List of generated equations
    gen_nlgs = []

    # Add non-modified equation
    if is_within_len(nlg, str_len) and \
            (not fix_len or is_fix_len(nlg, str_len)):
        gen_nlgs.append(nlg)

    # If shortest word is shorter than 2, pass the inserting
    if min([len(nlg[0][0]), len(nlg[0][1]), len(nlg[1][0]), len(nlg[1][1])]) < 2:
        return gen_nlgs

    # Calculate maximum size of insertion
    max_ins = int(math.floor(min((str_len[0] - len(nlg[0][0])) / (len(nlg[0][0]) - 1),
                                 (str_len[1] - len(nlg[0][1])) / (len(nlg[0][1]) - 1),
                                 (str_len[2] - len(nlg[1][0])) / (len(nlg[1][0]) - 1),
                                 (str_len[3] - len(nlg[1][1])) / (len(nlg[1][1]) - 1))))

    # Perform inserting
    if max_ins > 0:

        # List of all characters in current equation
        curr_chr = set(nlg[0][0] + nlg[0][1] + nlg[1][0] + nlg[1][1])

        # Get unique insertion chars
        chr_str_id = ord('0')
        ins_chr = []
        while len(ins_chr) < max_ins:
            if chr(chr_str_id) not in curr_chr:
                ins_chr.append(chr(chr_str_id))
            chr_str_id += 1

        # Generate all insertion chunks
        ins_chk = [[ins_chr[0]]]

        # Temporary list of previous length chunks
        prev_ins_chk = [[ins_chr[0]]]

        # Generate all the length of insert chunks
        for chr_id in range(1, max_ins):

            # Append a new character
            next_ins_chk = []
            for ins_chk in prev_ins_chk:
                for new_chr_id in range(len(set(ins_chk)) + 1):
                    next_ins_chk.append(ins_chk + [ins_chr[new_chr_id]])

            # Add to final chunk list
            ins_chk.extend(next_ins_chk)

            # Set to previous list for next iteration
            prev_ins_chk = next_ins_chk

        # Finally, perform insertion
        for ins_chk in ins_chk:

            # New equation
            inserted_nlg = AnalogyEquation()

            # Insert chunk between characters
            for ratio_idx, str_idx in product([0, 1], [0, 1]):

                # New string (list of symbols)
                new_string = []

                # For each character add insertion chunk after it
                for char in nlg[ratio_idx][str_idx][:-1]:
                    new_string.extend([char] + list(ins_chk))

                # Add last character to the end
                new_string.append(nlg[ratio_idx][str_idx][-1])

                # Set new string to object
                inserted_nlg[ratio_idx][str_idx] = new_string

            # Add to generated list
            if not fix_len or is_fix_len(inserted_nlg, str_len):
                gen_nlgs.append(inserted_nlg)

    return gen_nlgs


def gen_cast(nlg, str_len, fix_len=False):
    """
    Casting equations to maximum length
    :param nlg: an AnalogyEquation object
    :param str_len: length of all strings in generated equations
    :param fix_len: get only fixed length equations
    :return: list of casted equations
    """

    # Prefix character
    pre_chr = '<'

    # Suffix character
    suf_chr = '>'

    # List of generated equations
    gen_nlgs = []

    # Add non-modified equation
    if is_within_len(nlg, str_len) and \
            (not fix_len or is_fix_len(nlg, str_len)):
        gen_nlgs.append(nlg)

    # Loop with early breaks
    for pre_a in range(str_len[0] - len(nlg[0][0]) + 1):
        for suf_a in range(str_len[0] - len(nlg[0][0]) - pre_a + 1):
            for pre_b in range(str_len[1] - len(nlg[0][1]) + 1):
                for suf_b in range(str_len[1] - len(nlg[0][1]) - pre_b + 1):
                    for pre_c in range(max(0, pre_a - pre_b),
                                       str_len[2] - len(nlg[1][1]) + 1):
                        pre_d = pre_c + pre_b - pre_a
                        if pre_d + len(nlg[1][1]) > str_len[3]:
                            continue

                        for suf_c in range(str_len[2] - len(nlg[1][0]) - pre_c + 1):
                            suf_d = suf_c + suf_b - suf_a
                            if pre_d + len(nlg[1][1]) + suf_d > str_len[3]:
                                continue

                            # Sanity check (should always pass)
                            if pre_a >= 0 and pre_b >= 0 and pre_c >= 0 and pre_d >= 0 and suf_a >= 0 and suf_b >= 0 \
                                    and suf_c >= 0 and suf_d >= 0 and pre_d + len(nlg[1][1]) + suf_d <= str_len[3]:

                                # New equation
                                casted_nlg = AnalogyEquation()

                                casted_nlg[0][0] = ([pre_chr] * pre_a) + nlg[0][0] + ([suf_chr] * suf_a)
                                casted_nlg[0][1] = ([pre_chr] * pre_b) + nlg[0][1] + ([suf_chr] * suf_b)
                                casted_nlg[1][0] = ([pre_chr] * pre_c) + nlg[1][0] + ([suf_chr] * suf_c)
                                casted_nlg[1][1] = ([pre_chr] * pre_d) + nlg[1][1] + ([suf_chr] * suf_d)

                                # Add to generated list
                                if not fix_len or is_fix_len(casted_nlg, str_len):
                                    gen_nlgs.append(casted_nlg)

    return gen_nlgs


###############################################################################

def gen_fixed_len_nlgs_worker(nlg, method, str_len, fix_len):
    """
    Generate fixed length equations based on single equation
    :param nlg: an AnalogyEquation object
    :param method: forwarding method
    :param str_len: forwarding str_len
    :param fix_len: forwarding fix_len
    """

    gen_nlgs = []

    if method == 'r':  # Reduplicating
        gen_nlgs = gen_redup(nlg, str_len, fix_len)
    elif method == 'p':  # Permuting
        gen_nlgs = gen_permute(nlg, str_len, fix_len)
    elif method == 'm':  # Mirroring
        gen_nlgs = gen_mirror(nlg, str_len, fix_len)
    elif method == 'i':  # Inserting
        gen_nlgs = gen_insert(nlg, str_len, fix_len)
    elif method == 'c':  # Casting
        gen_nlgs = gen_cast(nlg, str_len, fix_len)

    return gen_nlgs


def gen_fixed_len_nlgs(nlgs, str_len, output=None, methods='pmric', var_length=False, verbose=False,
                       threads=multiprocessing.cpu_count()):
    """
    Generate fixed length equations based on nlgs and print out the output
    :param nlgs: list of AnalogyEquation objects
    :param str_len: length of strings in generated equations
    :param output: forwarding args.output
    :param methods: forwarding args.methods
    :param var_length: forwarding args.var_length
    :param verbose: forwarding args.verbose
    :param threads: forwarding args.threads
    """

    if verbose:
        print('Generate to length: {0} {1} {2} {3}'.format(*str_len), file=sys.stderr)
        print('Start generating equations', file=sys.stderr)

    # Multi-processing
    pool = None
    if threads > 1:
        pool = multiprocessing.Pool(threads)

    # If input AnalogyAlignment(s), strip into AnalogyEquation(s)
    if type(nlgs) == AnalogyAlignment:
        nlgs = [nlg_alg.nlg for nlg_alg in nlgs]

    gen_all_nlgs = []
    for nlg in nlgs:

        if verbose:
            print('Generate from: {}'.format(nlg), file=sys.stderr)

        # Initial list
        gen_prev_nlgs = [nlg]

        # Loop method
        for method_id, method in enumerate(list(methods)):

            # If last method, enable fix length filter
            if method_id == len(methods) - 1 and not var_length:
                fix_len = True
            else:
                fix_len = False

            # Print method id and name
            if verbose:
                if method == 'p':
                    print('{}. Permuting'.format(method_id + 1), file=sys.stderr)
                elif method == 'm':
                    print('{}. Mirroring'.format(method_id + 1), file=sys.stderr)
                elif method == 'r':
                    print('{}. Reduplicating'.format(method_id + 1), file=sys.stderr)
                elif method == 'i':
                    print('{}. Inserting'.format(method_id + 1), file=sys.stderr)
                elif method == 'c':
                    print('{}. Casting'.format(method_id + 1), file=sys.stderr)

            # Sent list to generation worker
            gen_tmp_nlgs = []

            if threads > 1:
                for gen_wrk_nlgs in pool.starmap(gen_fixed_len_nlgs_worker,
                                                 zip(gen_prev_nlgs, repeat(method), repeat(str_len), repeat(fix_len))):
                    gen_tmp_nlgs.extend(gen_wrk_nlgs)
            else:
                for prev_nlg in gen_prev_nlgs:
                    gen_tmp_nlgs.extend(gen_fixed_len_nlgs_worker(prev_nlg, method, str_len, fix_len))

            gen_prev_nlgs = gen_tmp_nlgs

            # Trigger garbage collection
            del gen_tmp_nlgs

            # Filter unique analogies
            gen_prev_nlgs = list(set(gen_prev_nlgs))

            # Print generation output
            if verbose:
                print('   - Numbers of equations: {}'.format(len(gen_prev_nlgs)), file=sys.stderr)

        gen_all_nlgs.extend(gen_prev_nlgs)

    # Filter unique generated analogies
    gen_all_nlgs = list(set(gen_all_nlgs))
    if verbose:
        print('Filtered generated analogies: {}'.format(len(gen_all_nlgs)), file=sys.stderr)

    # Create output stream (To file or stdout)
    out_file = None
    if output == 'sys.stdout':
        out_file = sys.stdout
    elif output is not None:
        if '.gz' in output:
            out_file = gzip.open(output, 'wt', encoding='utf8')
        else:
            out_file = open(output, 'w', encoding='utf8')

    # Print all out
    if out_file is not None:
        for nlg in gen_all_nlgs:
            print(nlg, file=out_file)

        # Close output file
        if out_file != sys.stdout:
            out_file.close()

    # Garbage collecting
    pool.close()
    del pool

    return gen_all_nlgs


###############################################################################

def main():
    # Start timer
    start_time = time.time()

    # Read equations
    nlgs = read_nlg_file(args.file)

    # Generate analogies & print output
    if len(args.length) == 1:
        args.length = (args.length[0],) * 4
    gen_fixed_len_nlgs(nlgs, args.length, args.output, args.methods, args.var_length, args.verbose, args.threads)

    if args.verbose:
        print('   - Execution time: {0:.2f} sec'.format(time.time() - start_time), file=sys.stderr)


if __name__ == '__main__':
    args = read_argv()
    main()
