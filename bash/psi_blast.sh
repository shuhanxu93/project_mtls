cd ../datasets/fasta
for file in *.fasta
do psiblast -query $file -evalue 0.01 -db ../../../my_database/uniprot_sprot.fasta -num_iterations 3 -out ../psiblast/$file.psiblast -out_ascii_pssm ../pssm/$file.pssm -num_threads 8
done
