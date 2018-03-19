import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import pandas as pd


np.set_printoptions(threshold=np.nan)


def main(dataset_file, output_file):

    model_list = ['FM_SVCrbf_unbalanced', 'FM_SVCrbf_balanced',
                  'FM_LinearSVC_unbalanced', 'FM_LinearSVC_balanced',
                  'FM_DT_unbalanced', 'FM_DT_balanced',
                  'FM_RF_unbalanced', 'FM_RF_balanced']
    accuracy_list = []
    recall_H_list = []
    recall_E_list = []
    recall_C_list = []
    precision_H_list = []
    precision_E_list = []
    precision_C_list = []
    f1_score_H_list =[]
    f1_score_E_list =[]
    f1_score_C_list =[]
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

        X_test = []

        for header in ids:
            pssm_filename = '../datasets/pssm/' + header + '.fasta.pssm'
            X_test.append(np.genfromtxt(pssm_filename, skip_header=3, skip_footer=5, usecols=range(22, 42))) # range(2, 22) for sm

        y_test = sec

        X_test_encoded, train_groups = encode_pssms(X_test, window_size) # train_groups is not needed for this script
        y_test_encoded = encode_targets(y_test)
        y_predicted = clf.predict(X_test_encoded)

        accuracy_list.append(accuracy_score(y_test_encoded, y_predicted))
        recall = recall_score(y_test_encoded, y_predicted, labels=[0, 1, 2], average=None)
        recall_H_list.append(recall[0])
        recall_E_list.append(recall[1])
        recall_C_list.append(recall[2])
        precision = precision_score(y_test_encoded, y_predicted, labels=[0, 1, 2], average=None)
        precision_H_list.append(precision[0])
        precision_E_list.append(precision[1])
        precision_C_list.append(precision[2])
        f1_scores = f1_score(y_test_encoded, y_predicted, labels=[0, 1, 2], average=None)
        f1_score_H_list.append(f1_scores[0])
        f1_score_E_list.append(f1_scores[1])
        f1_score_C_list.append(f1_scores[2])
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
                          'recall_H': recall_H_list,
                          'recall_E': recall_E_list,
                          'recall_C': recall_C_list,
                          'precision_H': precision_H_list,
                          'precision_E': precision_E_list,
                          'precision_C': precision_C_list,
                          'f1_score_H': f1_score_H_list,
                          'f1_score_E': f1_score_E_list,
                          'f1_score_C': f1_score_C_list,
                          'f1_macro': f1_macro_list,
                          'mcc': mcc_list,
                          'q3_ave': q3_ave_list}

    df = pd.DataFrame(evaluation_report)
    df.to_csv(output_file, sep='\t', encoding='utf-8', index=False)



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
    main('../datasets/new_proteins.txt', '../results/reports/report_fm_newproteins.csv') # modify the report file here(last argument)
