import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd

np.set_printoptions(threshold=np.nan)


def main(dataset_file):

    # parsing file
    ids, seq, sec = parse(dataset_file)

    # shuffle(random_state=0) and split sequences and structures into training(70%) and test sets(30%)
    X_train, X_test, y_train, y_test = train_test_split(seq, sec, test_size=0.3, random_state=0)

    # fragment X_train into sliding windows and get group labels

    y_train_encoded = encode_targets(y_train)

    dtc = tree.DecisionTreeClassifier()
    min_samples_split_range = [2, 4, 8, 16, 32, 64, 128]
    parameters = {'min_samples_split':min_samples_split_range}
    group_kfold = GroupKFold(n_splits=5)

    clf = GridSearchCV(dtc, parameters, n_jobs=-1, cv=group_kfold, verbose=2, error_score=np.NaN, return_train_score=False)

    test_windows = [11, 13, 15, 17, 19, 21, 23]

    for window_size in test_windows:
        X_train_fragmented, train_groups = fragment(X_train, window_size)
        X_train_encoded = encode_attributes(X_train_fragmented)
        clf.fit(X_train_encoded, y_train_encoded, groups=np.array(train_groups))
        df = pd.DataFrame(clf.cv_results_)
        output_file = '../results/seq_dt/seq_dt_none_' + str(window_size) + '.csv'
        df.to_csv(output_file, sep='\t', encoding='utf-8')




def parse(filename):
    """Parse though a protein sequence and secondary structure file
       and return lists of headers, sequences and structures"""

    headers = []
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

    half_window = window_size // 2

    fragmented_sequences = []
    groups = []
    for seq_idx, seq in enumerate(sequences):
        for amino_idx in range(len(seq)):
            if amino_idx < half_window:
                # N-terminal cases
                fragmented_sequences.append('0' * (half_window - amino_idx) + seq[:amino_idx + half_window + 1])
                groups.append(seq_idx)
            elif amino_idx >= len(seq) - half_window:
                # C-terminal cases
                fragmented_sequences.append(seq[amino_idx - half_window:] + '0' * (half_window - len(seq) + amino_idx + 1))
                groups.append(seq_idx)
            else:
                fragmented_sequences.append(seq[amino_idx - half_window:amino_idx + half_window + 1])
                groups.append(seq_idx)
    return fragmented_sequences, groups


def encode_attributes(fragmented_sequences):
    """Take a list of fragmented sequences and return a numpy array of one-hot encodings"""

    window_size = len(fragmented_sequences[0])

    encoded_sequences = np.zeros((len(fragmented_sequences), window_size, 20))
    for i in range(len(fragmented_sequences)):
        for j in range(window_size):
            encoded_sequences[i, j] = amino_code[fragmented_sequences[i][j]]
    encoded_sequences = encoded_sequences.reshape(len(fragmented_sequences), window_size * 20)

    return encoded_sequences


def encode_targets(structures):
    """Take a list of secondary structures and return a numpy array of class labels"""

    structures_str = ''.join(structures).translate(trans_table)
    encoded_structures = np.array(list(structures_str), dtype=int)
    return encoded_structures

# Create amino acid converter using dictionary
amino_code = {'A': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'R': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'N': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'D': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'C': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'Q': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'E': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'G': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'H': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'I': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'L': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'K': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
              'M': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
              'F': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
              'P': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
              'S': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
              'T': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
              'W': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
              'Y': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
              'V': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
              'X': np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
              'B': np.array([0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'Z': np.array([0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'J': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              '0': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}


# Create secondary structure converter using str.maketrans()
trans_table = str.maketrans("HEC", "012")
structure_name = np.array(['H', 'E', 'C'])


if __name__ == '__main__':
    main('../datasets/cas3.3line.txt')
