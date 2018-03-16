from Bio.PDB import PDBList

def main(filename):

    pdb_ids = parse(filename)

    pdb_ids_unique = []
    for pdb_id in pdb_ids:
        if pdb_id[:-1] not in pdb_ids_unique:
            pdb_ids_unique.append(pdb_id[:-1])

    fh = open("../datasets/pdb_ids.txt", 'w')

    pdbl = PDBList()
    for pdb_id in pdb_ids_unique[:150]:
        pdbl.retrieve_pdb_file(pdb_id, pdir='../datasets/pdb/')
        fh.write(pdb_id + '\n')

    fh.close()



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
    main('../datasets/cullpdb_pc10_res1.5_R0.3_d180316_chains1446.txt')
