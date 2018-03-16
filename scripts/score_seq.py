import itertools
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
import matplotlib.pyplot as plt


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
    X_test_fragmented, train_groups = fragment(X_test, window_size) # train_groups is not needed for this script

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
        sequence_fragmented, train_groups = fragment(sequence, window_size) # train_groups is not needed for this script
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

    plt.figure()
    plot_confusion_matrix(con_mat, classes=structure_name, normalize=True,
                      title='Normalized confusion matrix')

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
    main('../datasets/cas3.3line.txt', '../models/test_balanced.pkl') # modify model_file here (last 2 arguments)
