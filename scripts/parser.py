import numpy as np
from sklearn import svm

def main(dataset_file, window_size):

    print("Preprocessing training data...")

    ids, seq, sec = parse(dataset_file)

    seq_w = fragment(seq, window_size)

    features = encode_attributes(seq_w, window_size)[:10000]

    labels = encode_targets(sec)[:10000]

    print("Training...")

    clf = svm.SVC(C=1000)

    clf.fit(features, labels)

    print("Predicting...")

    predictions = clf.predict(features)

    accuracy = np.mean(predictions == labels)

    print("Accuracy =", accuracy)

def parse(filename):
    """Parse though a protein sequence and secondary structure file
       and return lists of headers, sequences and structures"""
    headers =[]
    sequences = []
    structures = []

    with open(filename) as fh:
        while True:
            header = fh.readline().rstrip()[1:]
            sequence = fh.readline().rstrip()
            structure = fh.readline().rstrip()
            if len(structure) == 0:
                break
            headers.append(header)
            sequences.append(sequence)
            structures.append(structure)
    return headers, sequences, structures

def fragment(sequences, window_size):
    """Take a list of protein sequences and return a list of window-sized sequences"""

    # append 0s to heads and tails of sequences
    half_window = window_size // 2
    adjusted_sequences = ['0'*half_window + i + '0'*half_window for i in sequences]

    # breakdown sequences into window-sized pieces
    fragmented_sequences = []
    for seq in adjusted_sequences:
        for j in range(half_window, len(seq) - half_window):
            fragmented_sequences.append(seq[j-half_window:j+half_window+1])

    return fragmented_sequences

def encode_attributes(fragmented_sequences, window_size):
    """Convert a list of fragmented sequences to a matrix of one-hot encoded vectors"""

    encoded_vectors = np.zeros((len(fragmented_sequences), amino_size * window_size), dtype=int)
    for i in range(len(fragmented_sequences)):
        for j in range(window_size):
            encoded_vectors[i, j * amino_size + amino2num[fragmented_sequences[i][j]]] = 1

    return encoded_vectors

def encode_targets(structures):
    """Convert a list of secondary structures to a vector of class labels"""
    labels_str = ''.join(structures).translate(trans_table)
    labels_list = list(map(int, list(labels_str)))
    class_labels = np.array(labels_list)
    return class_labels

# Create amino acid converter using dictionary
amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '0']
amino_size = len(amino_acids)
amino_num = [i for i in range(amino_size)]
amino2num = dict((x, y) for x, y in zip(amino_acids, amino_num))

# Create secondary structure converter using str.maketrans()
sec_str = "HEC"
sec_num = "012"
trans_table = str.maketrans(sec_str, sec_num)

sec_name = np.array(list(sec_str))

if __name__ == '__main__':
    main('../datasets/cas3_removed.txt', 17)
