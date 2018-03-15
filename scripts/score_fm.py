import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


np.set_printoptions(threshold=np.nan)


def main(dataset_file, model_file):

    # load window_size and predictor from model_file
    data = joblib.load(model_file)
    window_size = data['window_size']
    clf = data['clf']

    # parsing file
    ids, seq, sec = parse(dataset_file)

    # shuffle(random_state=0) and split sequences and structures into training(70%) and test sets(30%)
    X_train, X_test, y_train, y_test = train_test_split(seq, sec, test_size=0.3, random_state=0)

    # fragment X_train into sliding windows and get group labels
    X_test_fragmented, dummy = fragment(X_test, window_size) # dummy is not needed for this script

    X_test_encoded = encode_attributes(X_test_fragmented)
    y_test_encoded = encode_targets(y_test)
    y_predicted = clf.predict(X_test_encoded)

    accuracy = accuracy_score(y_test_encoded, y_predicted)
    con_mat = confusion_matrix(y_test_encoded, y_predicted, labels=[0, 1, 2])
    recall = recall_score(y_test_encoded, y_predicted, labels=[0, 1, 2], average=None)
    precision = precision_score(y_test_encoded, y_predicted, labels=[0, 1, 2], average=None)
    f1_scores = f1_score(y_test_encoded, y_predicted, labels=[0, 1, 2], average=None)
    f1_macro = f1_score(y_test_encoded, y_predicted, average='macro')
    mcc = matthews_corrcoef(y_test_encoded, y_predicted)

    q3_scores = np.zeros(len(X_test))
    for index in range(len(X_test)):
        sequence = [X_test[index]]
        structure = [y_test[index]]
        sequence_fragmented, dummy = fragment(sequence, window_size)
        sequence_encoded = encode_attributes(sequence_fragmented)
        structure_encoded = encode_targets(structure)
        q3_scores[index] = clf.score(sequence_encoded, structure_encoded)
    q3_ave = np.mean(q3_scores)

    print("accuracy =", accuracy)
    print("confusion matrix=", con_mat)
    print("recall =", recall)
    print("precision =", precision)
    print("f1 scores =", f1_scores)
    print("f1 macro = ", f1_macro)
    print("Matthews correlation coefficient (MCC) =", mcc)
    print("Q3 score =", q3_ave)




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
    main('../datasets/cas3.3line.txt', '../models/test_balanced.pkl') # modify model_file here (last 2 arguments)