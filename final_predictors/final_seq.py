"""This script predicts protein secondary structure from primary sequence"""

from sys import argv
import numpy as np
from sklearn.externals import joblib

def main(model_file):
    """Main function of script. Uses the model_file to predict the secondary
       structure of input fasta file. Predicted secondary structure is printed
       to screen and saved in an output_file."""

    # unpack arguments
    _, input_file, output_file = argv[:3]

    # load window_size and predictor from model_file
    data = joblib.load(model_file)
    window_size = data['window_size']
    clf = data['clf']

    # parsing input_file
    ids, seq = parse(input_file)

    filehandle = open(output_file, 'w')
    for index, _ in enumerate(ids):
        sequence = [seq[index]]
        sequence_fragmented, _ = fragment(sequence, window_size)
        sequence_encoded = encode_attributes(sequence_fragmented)
        prediction = ''.join(STRUCTURE_NAME[clf.predict(sequence_encoded)])
        filehandle.write('>' + ids[index] + '\n')
        filehandle.write(seq[index] + '\n')
        filehandle.write(prediction + '\n')
        print('>' + ids[index])
        print(seq[index])
        print(prediction)
    filehandle.close()


def parse(filename):
    """Parse though a protein sequence and secondary structure file
       and return lists of headers, sequences and structures"""

    headers = []
    sequences = []

    with open(filename) as filehandle:
        while True:
            header = filehandle.readline().rstrip()[1:]
            sequence = filehandle.readline().rstrip()
            if len(sequence) == 0:
                break
            headers.append(header)
            sequences.append(sequence)
    return headers, sequences


def fragment(sequences, window_size):
    """Take a list of protein sequences and return a list of window-sized
       sequences"""

    half_window = window_size // 2

    fragmented_sequences = []
    groups = []
    for seq_idx, seq in enumerate(sequences):
        for amino_idx in range(len(seq)):
            if amino_idx < half_window:
                # N-terminal cases
                fragmented_sequences.append('0' * (half_window - amino_idx) +
                                            seq[:amino_idx + half_window + 1])
                groups.append(seq_idx)
            elif amino_idx >= len(seq) - half_window:
                # C-terminal cases
                fragmented_sequences.append(seq[amino_idx - half_window:] +
                                            '0' * (half_window - len(seq) +
                                                   amino_idx + 1))
                groups.append(seq_idx)
            else:
                fragmented_sequences.append(seq[amino_idx -
                                                half_window:amino_idx +
                                                half_window + 1])
                groups.append(seq_idx)
    return fragmented_sequences, groups


def encode_attributes(fragmented_sequences):
    """Take a list of fragmented sequences and return a numpy array of one-hot
       encodings"""

    window_size = len(fragmented_sequences[0])

    encoded_sequences = np.zeros((len(fragmented_sequences), window_size, 20))
    for i, _ in enumerate(fragmented_sequences):
        for j in range(window_size):
            encoded_sequences[i, j] = AMINO_CODE[fragmented_sequences[i][j]]
    encoded_sequences = encoded_sequences.reshape(len(fragmented_sequences),
                                                  window_size * 20)

    return encoded_sequences


# Create amino acid converter using dictionary
AMINO_CODE = {'A': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
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
              'X': np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                             0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
              'B': np.array([0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'Z': np.array([0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              'J': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              '0': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}


STRUCTURE_NAME = np.array(['H', 'E', 'C'])


if __name__ == '__main__':
    main('./seq_SVCrbf_unbalanced.pkl')
