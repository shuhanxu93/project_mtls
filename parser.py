import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelBinarizer
np.set_printoptions(threshold=np.nan)

def parser(filename):
    '''Parse though a protein sequence file and return lists of headers, sequences and structures'''
    headers =[]
    sequences = []
    structures = []

    with open(filename) as fh:
        while True:
            header = fh.readline().rstrip()
            sequence = fh.readline().rstrip()
            structure = fh.readline().rstrip()
            if len(structure) == 0:
                break
            headers.append(header)
            sequences.append(sequence)
            structures.append(structure)
    return headers, sequences, structures

def create_dataset(raw_sequences, window_size):
    """Create a matrix of n_samples * n_features. n_features = 21 amino acids * window_size."""

    # append Xs to heads and tails of sequences
    half_window = window_size // 2
    adjusted_sequences = ['X'*half_window + i + 'X'*half_window for i in raw_sequences]

    # breakdown sequences into window-sized pieces
    fragmented_sequences = []

    for seq in adjusted_sequences:
        for j in range(half_window, len(seq) - half_window):
            fragmented_sequences.append(seq[j-half_window:j+half_window+1])

    # Create amino acid converter
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    amino_num = [i for i in range(len(amino_acids))]
    amino2num = dict((x, y) for x, y in zip(amino_acids, amino_num))

    # convert sequences to array of binaries
    data_array = np.zeros((len(fragmented_sequences), 21 * window_size), dtype=int)
    for i in range(len(fragmented_sequences)):
        for j in range(len(fragmented_sequences[i])):
            data_array[i, j * 21 + amino2num[fragmented_sequences[i][j]]] = 1

    return data_array

def create_groundtruth(raw_structures):
    """Create a vector of n_samples of secondary structures"""
    long_string = ''.join(raw_structures)
    structure_array = np.array(list(long_string))
    return structure_array

headers, sequences, structures = parser('test_set.txt')

print(create_dataset(sequences, 7))
