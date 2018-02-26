import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def main(dataset_file, window_size):

    print("Preprocessing training data...")

    ids, seq, sec = parse(dataset_file)

    seq_w = fragment(seq, window_size)

    features = encode_attributes(seq_w, window_size)

    labels = encode_targets(sec)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4)

    title = "Learning Curves (SVM, RBF kernel, C=1.0, gamma='auto', window_size=17)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    estimator = svm.SVC()
    train_sizes = np.linspace(.05, 1.0, 10)
    plot_learning_curve(estimator, title, X_train, y_train, (0.30, 1.01), cv=cv, n_jobs=4, train_sizes=train_sizes)

    plt.show()

    '''
    print("Training...")

    clf = svm.SVC(C=1000)

    clf.fit(features, labels)

    print("Predicting...")

    predictions = clf.predict(features)

    accuracy = np.mean(predictions == labels)

    print("Accuracy =", accuracy)
    '''

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


# code from sklearn
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

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
