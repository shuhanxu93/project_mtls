def encode(fragmented_sequences, window_size):
    """Take a list of fragmented sequences and return a numpy array of one-hot encodings"""

    encoded_sequences = np.zeros((len(fragmented_sequences), window_size, 21))
    for i in range(len(fragmented_sequences)):
        for j in range(window_size):
            encoded_sequences[i, j] = amino_code[fragmented_sequences[i][j]]
    encoded_sequences = encoded_sequences.reshape(len(fragmented_sequences), window_size * 21)

    return encoded_sequences


def encode(fragmented_sequences, window_size):
    """Take a list of fragmented sequences and return a numpy array of one-hot encodings"""

    encoded_sequences = np.zeros((len(fragmented_sequences), window_size, 21))
    for i in range(len(fragmented_sequences)):
        for j in range(window_size):
            encoded_sequences[i, j] = amino_code[fragmented_sequences[i][j]]
    encoded_sequences = encoded_sequences.reshape(len(fragmented_sequences), window_size * 21))

    return encoded_sequences
