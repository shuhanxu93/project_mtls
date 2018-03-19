"""This script predicts protein secondary structure from PSSM"""

from sys import argv
import numpy as np
from sklearn.externals import joblib

def main(model_file):
    """Main function of script. Uses the model_file to predict the secondary
       structure of input PSSM. Predicted secondary structure is printed to
       screen and saved in an output_file."""

    # unpack arguments from argv
    _, input_file, output_file = argv[:3]

    # load window_size and predictor from model_file
    data = joblib.load(model_file)
    window_size = data['window_size']
    clf = data['clf']

    sequence = ''.join(np.genfromtxt(input_file, dtype=str, skip_header=3,
                                     skip_footer=5, usecols=1))
    pssm = [np.genfromtxt(input_file, skip_header=3, skip_footer=5,
                          usecols=range(22, 42))]

    filehandle = open(output_file, 'w')
    pssm_encoded, _ = encode_pssms(pssm, window_size)
    prediction = ''.join(STRUCTURE_NAME[clf.predict(pssm_encoded)])
    filehandle.write('>' + input_file + '\n')
    filehandle.write(sequence + '\n')
    filehandle.write(prediction + '\n')
    print('>' + input_file)
    print(sequence)
    print(prediction)
    filehandle.close()


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
        padded_pssm = np.vstack([np.zeros((half_window, 20)), scaled_pssm,
                                 np.zeros((half_window, 20))])
        for i in range(seq_len):
            training_array[i] = padded_pssm[i:i + window_size]
            groups.append(pssm_idx)
        arrays_list.append(training_array.reshape(seq_len, window_size * 20))
    return np.vstack(arrays_list), groups


STRUCTURE_NAME = np.array(['H', 'E', 'C'])


if __name__ == '__main__':
    main('./FM_SVCrbf_balanced.pkl')
