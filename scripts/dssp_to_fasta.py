import numpy as np
from re import sub

fh = open("../datasets/pdb_ids.txt", 'r')
pdb_ids = fh.read().splitlines()
fh.close()

ids = []
sequences = []
structures = []
for pdb_id in pdb_ids:
    dssp_filename = '../datasets/dssp/pdb' + pdb_id.lower() + '.ent.dssp'
    check = np.genfromtxt(dssp_filename, skip_header=28, usecols=1)
    if not np.isnan(check).any(): # check if structure is one single sequence
        ids.append(pdb_id.lower())
        seq = ''.join(np.genfromtxt(dssp_filename, skip_header=28, usecols=1,
                      dtype=str, delimiter=[13,1]).tolist())
        seq = sub("[a-z]", 'C', seq) # convert lowercase amino acid to cysteine
        sequences.append(seq)
        sec = ''.join(np.genfromtxt(dssp_filename, skip_header=28, usecols=1,
                      dtype=str, delimiter=[16,1]).tolist())
        sec = sec.replace(' ', 'L') # convert space to 'L'
        structures.append(sec)

for index in range(len(ids)):
    filename = '../datasets/new_proteins_fasta/' + ids[index] + '.fasta'
    writefile = open(filename, 'w')
    writefile.write('>' + ids[index] + '\n')
    writefile.write(sequences[index] + '\n')
    writefile.close()

writefile = open('../datasets/new_proteins.txt', 'w')
for index in range(len(ids)):
    writefile.write('>' + ids[index] + '\n')
    writefile.write(sequences[index] + '\n')
    writefile.write(structures[index] + '\n')
writefile.close()
