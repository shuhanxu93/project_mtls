import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score

np.set_printoptions(threshold=np.nan)


def main(dataset_file):

    # parsing file
    ids, seq, sec = parse(dataset_file)

    for index in range(len(ids)):
        filename = '../datasets/fasta/' + ids[index] + '.fasta'
        writefile = open(filename, 'w')
        writefile.write('>' + ids[index] + '\n')
        writefile.write(seq[index] + '\n')
        writefile.close()


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

if __name__ == '__main__':
    main('../datasets/cas3.3line.txt')
