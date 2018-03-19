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

    model_list = ['seq_SVCrbf_unbalanced', 'seq_SVCrbf_balanced',
                  'seq_LinearSVC_unbalanced', 'seq_LinearSVC_balanced',
                  'seq_DT_unbalanced', 'seq_DT_balanced',
                  'seq_RF_unbalanced', 'seq_RF_balanced']
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

        X_test = seq
        y_test = sec

        # fragment X_test into sliding windows
        X_test_fragmented, train_groups = fragment(X_test, window_size) # train_groups is not needed for this script

        X_test_encoded = encode_attributes(X_test_fragmented)
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
            sequence = [X_test[index]]
            structure = [y_test[index]]
            sequence_fragmented, train_groups = fragment(sequence, window_size) # train_groups is not needed for this script
            sequence_encoded = encode_attributes(sequence_fragmented)
            structure_encoded = encode_targets(structure)
            q3_scores[index] = clf.score(sequence_encoded, structure_encoded)
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
    main('../datasets/new_proteins.txt', '../results/reports/report_seq_newproteins.csv') # modify the report file here(last argument)
