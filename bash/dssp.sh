cd ../datasets/pdb
for file in *.ent
do dssp -i $file -o ../dssp/$file.dssp
done
