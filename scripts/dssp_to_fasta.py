import numpy as np

fh = open("../datasets/pdb_ids.txt", 'r')

pdb_ids = fh.read().splitlines()

fh.close()

seq = []

sec = []

number = 0
for pdb_id in pdb_ids:
    dssp_filename = '../datasets/dssp/pdb' + pdb_id.lower() + '.ent.dssp'
    check = np.genfromtxt(dssp_filename, skip_header=28, usecols=1)
    print(pdb_id)
    if not np.isnan(check).any():
        seq.append(pdb_id)
print(seq)
