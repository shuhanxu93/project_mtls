import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

np.set_printoptions(threshold=np.nan)

def main(dataset_file, window_size):

    # parsing file
    ids, seq, sec = parse(dataset_file)

    # shuffle(random_state=0) and split sequences and structures into training(70%) and test sets(30%)
    X_train, X_test, y_train, y_test = train_test_split(seq, sec, test_size=0.3, random_state=0)

    # creates 5 training datasets using KFold.
    kf = KFold(n_splits=5)
    kf_list = list(kf.split(X_train))

    X_train_0, X_val_0 = [X_train[i] for i in kf_list[0][0]], [X_train[i] for i in kf_list[0][1]]
    y_train_0, y_val_0 = [y_train[i] for i in kf_list[0][0]], [y_train[i] for i in kf_list[0][1]]

    X_train_1, X_val_1 = [X_train[i] for i in kf_list[1][0]], [X_train[i] for i in kf_list[1][1]]
    y_train_1, y_val_1 = [y_train[i] for i in kf_list[1][0]], [y_train[i] for i in kf_list[1][1]]

    X_train_2, X_val_2 = [X_train[i] for i in kf_list[2][0]], [X_train[i] for i in kf_list[2][1]]
    y_train_2, y_val_2 = [y_train[i] for i in kf_list[2][0]], [y_train[i] for i in kf_list[2][1]]

    X_train_3, X_val_3 = [X_train[i] for i in kf_list[3][0]], [X_train[i] for i in kf_list[3][1]]
    y_train_3, y_val_3 = [y_train[i] for i in kf_list[3][0]], [y_train[i] for i in kf_list[3][1]]

    X_train_4, X_val_4 = [X_train[i] for i in kf_list[4][0]], [X_train[i] for i in kf_list[4][1]]
    y_train_4, y_val_4 = [y_train[i] for i in kf_list[4][0]], [y_train[i] for i in kf_list[4][1]]

    X_train_kfold = [X_train_0, X_train_1, X_train_2, X_train_3, X_train_4]
    X_val_kfold = [X_val_0, X_val_1, X_val_2, X_val_3, X_val_4]

    y_train_kfold = [y_train_0, y_train_1, y_train_2, y_train_3, y_train_4]
    y_val_kfold = [y_val_0, y_val_1, y_val_2, y_val_3, y_val_4]

    # preprocess datasets
    X_train_datasets = attributes_preprocess(X_train_kfold, window_size)
    X_val_datasets = attributes_preprocess(X_val_kfold, window_size)

    y_train_datasets = targets_preprocess(y_train_kfold)
    y_val_datasets = targets_preprocess(y_val_kfold)









    '''
    for train_index, test_index in kf.split(X_train):
        train_folds.append(train_index)
        test_folds.append(test_index)

    print(train_folds)
    print(test_folds)
    '''



'''
    print(sel_train)
    # split training set into 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
'''
#    for train_index, test_index in kf.split(sel_train):
#        print("TRAIN:", train_index, "TEST:", test_index)












'''
    print("Training...")

    clf = svm.SVC(C=1000,gamma=0.003, cache_size=7000, verbose=2)

    clf.fit(X_train, y_train)

    print("Predicting...")

    predictions = clf.predict(X_train) # need to change to X_test

    accuracy = np.mean(predictions == y_train) # need to change to y_test

    print("Accuracy =", accuracy)
'''


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
    for seq in sequences:
        for index in range(len(seq)):
            if index < half_window:
                # N-terminal cases
                fragmented_sequences.append('0' * (half_window - index) + seq[:index + half_window + 1])
            elif index >= len(seq) - half_window:
                # C-terminal cases
                fragmented_sequences.append(seq[index - half_window:] + '0' * (half_window - len(seq) + index + 1))
            else:
                fragmented_sequences.append(seq[index - half_window:index + half_window + 1])
    return fragmented_sequences


def attributes_preprocess(attributes_kfold, window_size):
    """Take a list of K-Fold lists of sequences and return a list of K-Fold numpy arrays of one-hot encodings"""

    # fragment sequences in K-Fold
    kfold_fragmented = []
    for fold in attributes_kfold:
        kfold_fragmented.append(fragment(fold, window_size))

    # encode fragmented sequences in K-Fold
    kfold_encoded = []
    for fold in kfold_fragmented:
        fold_encoded = np.zeros((len(fold), window_size, 21))
        for i in range(len(fold)):
            for j in range(window_size):
                fold_encoded[i, j] = amino_code[fold[i][j]]
        kfold_encoded.append(fold_encoded.reshape(len(fold), window_size * 21))

    return kfold_encoded


def targets_preprocess(targets_kfold):
    """Take a list of K-Fold lists of secondary structures and return a list of K-Fold numpy arrays of class labels"""

    kfold_labelled = []
    for fold in targets_kfold:
        fold_str = ''.join(fold).translate(trans_table)
        kfold_labelled.append(np.array(list(fold_str), dtype=int))
    return kfold_labelled

# Create amino acid converter using dictionary
amino_code = {'A': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'R': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'N': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'D': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'C': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'Q': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'E': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'G': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'H': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'I': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'L': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'K': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'M': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
              'F': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
              'P': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
              'S': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
              'T': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
              'W': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
              'Y': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
              'V': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
              'X': np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0]),
              'B': np.array([0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'Z': np.array([0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'J': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              '0': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])}


# Create secondary structure converter using str.maketrans()
trans_table = str.maketrans("HEC", "012")
structure_name = np.array(['H', 'E', 'C'])


if __name__ == '__main__':
    main('../datasets/cas3.3line.txt', 17)
