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
import pandas as pd


np.set_printoptions(threshold=np.nan)


def main(dataset_file, output_file):

    model_list = ['test_fm_none', 'test_fm_balanced']
    accuracy_list = []
    con_mat_list = []
    recall_list = []
    precision_list = []
    f1_scores_list =[]
    f1_macro_list = []
    mcc_list = []
    q3_ave_list = []

    for model in model_list:
        # load window_size and predictor from model_file
        model_file = '../models/' + model + '.pkl'
        data = joblib.load(model_file)
        window_size = data['window_size']
        clf = data['clf']

        # parsing file
        ids, seq, sec = parse(dataset_file)

        # shuffle(random_state=0) and split sequences and structures into training(70%) and test sets(30%)
        X_train, X_test, y_train, y_test = train_test_split(ids, sec, test_size=0.3, random_state=0)

        index = X_test.index('1wfbb-1-AUTO.1') # remove 1wfbb-1-AUTO.1 as it has no hit in psiblast
        del X_test[index]
        del y_test[index]

        pssms = []

        for header in X_test:
            pssm_filename = '../datasets/pssm/' + header + '.fasta.pssm'
            pssms.append(np.genfromtxt(pssm_filename, skip_header=3, skip_footer=5, usecols=range(22, 42))) # range(2, 22) for sm

        X_test = pssms

        X_test_encoded, train_groups = encode_pssms(X_test, window_size) # train_groups is not needed for this script
        y_test_encoded = encode_targets(y_test)
        y_predicted = clf.predict(X_test_encoded)

        accuracy_list.append(accuracy_score(y_test_encoded, y_predicted))
        con_mat_list.append(confusion_matrix(y_test_encoded, y_predicted, labels=[0, 1, 2]))
        recall_list.append(recall_score(y_test_encoded, y_predicted, labels=[0, 1, 2], average=None))
        precision_list.append(precision_score(y_test_encoded, y_predicted, labels=[0, 1, 2], average=None))
        f1_scores_list.append(f1_score(y_test_encoded, y_predicted, labels=[0, 1, 2], average=None))
        f1_macro_list.append(f1_score(y_test_encoded, y_predicted, average='macro'))
        mcc_list.append(matthews_corrcoef(y_test_encoded, y_predicted))

        q3_scores = np.zeros(len(X_test))
        for index in range(len(X_test)):
            pssm = [X_test[index]]
            structure = [y_test[index]]
            pssm_encoded, train_groups = encode_pssms(pssm, window_size) # train_groups is not needed for this script
            structure_encoded = encode_targets(structure)
            q3_scores[index] = clf.score(pssm_encoded, structure_encoded)
        q3_ave_list.append(np.mean(q3_scores))


    evaluation_report = { 'model': model_list,
                          'accuracy': accuracy_list,
                          'con_mat': con_mat_list,
                          'recall': recall_list,
                          'precision': precision_list,
                          'f1_scores': f1_scores_list,
                          'f1_macro': f1_macro_list,
                          'mcc': mcc_list,
                          'q3_ave': q3_ave_list}

    df = pd.DataFrame(evaluation_report)
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
    main('../datasets/cas3.3line.txt', '../results/reports/report2.csv') # modify model_file here (last 2 arguments)
