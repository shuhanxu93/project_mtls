import numpy as np
from sklearn import svm

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
    """Create a matrix of n_samples * n_features. n_features = amino_size * window_size."""

    # append Xs to heads and tails of sequences
    half_window = window_size // 2
    adjusted_sequences = ['J'*half_window + i + 'J'*half_window for i in raw_sequences]

    # breakdown sequences into window-sized pieces
    fragmented_sequences = []

    for seq in adjusted_sequences:
        for j in range(half_window, len(seq) - half_window):
            fragmented_sequences.append(seq[j-half_window:j+half_window+1])

    # Create amino acid converter
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'J', 'X', 'B', 'Z']
    amino_size = len(amino_acids)
    amino_num = [i for i in range(amino_size)]
    amino2num = dict((x, y) for x, y in zip(amino_acids, amino_num))

    # convert sequences to array of binaries
    data_array = np.zeros((len(fragmented_sequences), amino_size * window_size), dtype=int)
    for i in range(len(fragmented_sequences)):
        for j in range(len(fragmented_sequences[i])):
            data_array[i, j * amino_size + amino2num[fragmented_sequences[i][j]]] = 1

    return data_array

def create_groundtruth(raw_structures):
    """Create a vector of n_samples of secondary structures"""
    long_string = ''.join(raw_structures)
    structure_array = np.array(list(long_string))
    return structure_array

headers, sequences, structures = parser('cas3.3line.txt')


questions = create_dataset(sequences, 17)[:11000]
answers = create_groundtruth(structures)[:11000]

print("preprocessing done")

clf = svm.SVC(C=1000)
clf.fit(questions, answers)

predictions = clf.predict(questions)

accuracy = np.mean(predictions == answers)

print(accuracy)
