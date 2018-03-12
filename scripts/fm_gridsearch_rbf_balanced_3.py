import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd

np.set_printoptions(threshold=np.nan)


def main(dataset_file):

    # parsing file
    ids, seq, sec = parse(dataset_file)

    # shuffle(random_state=0) and split sequences and structures into training(70%) and test sets(30%)
    X_train, X_test, y_train, y_test = train_test_split(ids, sec, test_size=0.3, random_state=0)

    pssms = []

    for header in X_train:
        pssm_filename = '../datasets/pssm/' + header + '.fasta.pssm'
        pssms.append(np.genfromtxt(pssm_filename, skip_header=3, skip_footer=5, usecols=range(22, 42))) # range(2, 22) for sm

    X_train = pssms

    y_train_encoded = encode_targets(y_train)

    svc = svm.SVC(kernel='rbf', cache_size=5000, class_weight='balanced')
    C_range = np.power(2, np.linspace(-5, 15, 11)).tolist()
    gamma_range = np.power(2, np.linspace(-15, 3, 10)).tolist()
    parameters = {'C':C_range, 'gamma':gamma_range}
    group_kfold = GroupKFold(n_splits=5)
    scoring = ['accuracy', 'f1_macro']

    clf = GridSearchCV(svc, parameters, scoring=scoring, n_jobs=-1, cv=group_kfold, verbose=2, error_score=np.NaN, return_train_score=False)

    test_windows = [21, 23]

    for window_size in test_windows:
        X_train_encoded, train_groups = encode_pssms(X_train, window_size)
        clf.fit(X_train_encoded, y_train_encoded, groups=np.array(train_groups))
        df = pd.DataFrame(clf.cv_results_)
        output_file = '../results/fm_rbf_balanced_' + str(window_size) + '.csv'
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
    main('../datasets/cas3.3line.txt')
