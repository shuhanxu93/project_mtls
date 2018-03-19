import itertools
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


np.set_printoptions(threshold=np.nan)


def main(dataset_file, model_file):

    # load window_size and predictor from model_file
    data = joblib.load(model_file)
    window_size = data['window_size']
    clf = data['clf']

    # parsing file
    ids, seq, sec = parse(dataset_file)

    X_test = []

    for header in ids:
        pssm_filename = '../datasets/pssm/' + header + '.fasta.pssm'
        X_test.append(np.genfromtxt(pssm_filename, skip_header=3, skip_footer=5, usecols=range(22, 42))) # range(2, 22) for sm

    y_test = sec

    X_test_encoded, train_groups = encode_pssms(X_test, window_size) # train_groups is not needed for this script
    y_test_encoded = encode_targets(y_test)
    y_predicted = clf.predict(X_test_encoded)

    con_mat = confusion_matrix(y_test_encoded, y_predicted, labels=[0, 1, 2])

    plt.figure()
    plot_confusion_matrix(con_mat, classes=structure_name, normalize=True,
                      title='Normalized confusion matrix for FM_SVCrbf_balanced (New Proteins)')
    plt.tight_layout()
    plt.show()



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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Create secondary structure converter using str.maketrans()
trans_table = str.maketrans("HEC", "012")
structure_name = np.array(['H', 'E', 'C'])


if __name__ == '__main__':
    main('../datasets/new_proteins.txt', '../models/FM_SVCrbf_balanced.pkl') # modify model_file here (last 2 arguments)
