import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

np.set_printoptions(threshold=np.nan)


def main(dataset_file, window_size, model_name):

    # parsing file
    ids, seq, sec = parse(dataset_file)

    # shuffle(random_state=0) and split sequences and structures into training(70%) and test sets(30%)
    X_train, X_test, y_train, y_test = train_test_split(ids, sec, test_size=0.3, random_state=0)

    pssms = []

    for header in X_train:
        pssm_filename = '../datasets/pssm/' + header + '.fasta.pssm'
        pssms.append(np.genfromtxt(pssm_filename, skip_header=3, skip_footer=5, usecols=range(22, 42))) # range(2, 22) for sm

    X_train = pssms

    X_train_encoded, train_groups = encode_pssms(X_train, window_size)
    y_train_encoded = encode_targets(y_train)

    clf = svm.SVC(C=8.0, gamma=0.125000, cache_size=5000, class_weight='balanced') # modify the hyper-parameters here
    clf.fit(X_train_encoded, y_train_encoded)

    data = {'model_name': model_name, 'window_size': window_size, 'clf': clf}
    model_file = '../models/' + model_name + '.pkl'
    joblib.dump(data, model_file)



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


def encode_pssms(pssms_list, window_size):
    """Take a list of pssm numpy arrays
       and return a numpy array of n_samples * n_features for training"""

    half_window = window_size // 2

    arrays_list = []
    groups = []
    for pssm_idx, pssm in enumerate(pssms_list):
        seq_len = len(pssm)
        training_array = np.zeros((seq_len, window_size, 20))
        scaled_pssm = pssm / 100
        padded_pssm = np.vstack([np.zeros((half_window, 20)), scaled_pssm, np.zeros((half_window, 20))])
        for i in range(seq_len):
            training_array[i] = padded_pssm[i:i + window_size]
            groups.append(pssm_idx)
        arrays_list.append(training_array.reshape(seq_len, window_size * 20))
    return np.vstack(arrays_list), groups


def encode_targets(structures):
    """Take a list of secondary structures and return a numpy array of class labels"""

    structures_str = ''.join(structures).translate(trans_table)
    encoded_structures = np.array(list(structures_str), dtype=int)
    return encoded_structures


# Create secondary structure converter using str.maketrans()
trans_table = str.maketrans("HEC", "012")
structure_name = np.array(['H', 'E', 'C'])


if __name__ == '__main__':
    main('../datasets/cas3.3line.txt', 15, 'test_fm_balanced') # modify window size and model name here (last 2 arguments)
