from Bio.PDB import PDBList


def main(filename):

    pdb_ids = parse(filename)

    pdb_ids_unique = []

    for pdb_id in pdb_ids:
        if pdb_id[:-1] not in pdb_ids_unique:
            pdb_ids_unique.append(pdb_id[:-1])
    
    pdbl = PDBList()
    for pdb_id in pdb_ids_unique[:75]:
        pdbl.retrieve_pdb_file(pdb_id, pdir='../datasets/pdb')
        


def parse(filename):

    pdb_ids = []

    with open(filename) as fh:
        fh.readline()
        while True:
            line = fh.readline()
            if len(line) == 0:
                break
            pdb_ids.append(line.split()[0])
    return pdb_ids


if __name__ == '__main__':
    main('../datasets/cullpdb_pc5_res1.5_R0.3_d180315_chains1415.txt')
