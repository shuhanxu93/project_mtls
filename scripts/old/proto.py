import numpy as np
np.set_printoptions(threshold=np.nan)

pssm_filename = '../datasets/pssm/1ubdc-1-AS.fasta.pssm'
pssm = np.genfromtxt(pssm_filename, skip_header=3, skip_footer=5, usecols=range(22, 42))

window_size = 3
half_window = window_size // 2


# new_pssm = np.concatenate(((np.zeros((half_window, 20))), pssm), axis=0)
seq_len = len(pssm)
training_array = np.zeros((seq_len, window_size, 20))
padded_pssm = np.vstack([np.zeros((half_window, 20)), pssm, np.zeros((half_window, 20))])
for i in range(seq_len):
    training_array[i] = padded_pssm[i:i + window_size]
x = training_array.reshape(seq_len, window_size * 20)
print(training_array)
