cd ../datasets/fasta
for file in *.fasta
do psiblast -query $file -evalue 0.001 -db ../../../my_database/uniref50.fasta -num_iterations 3 -out ../psiblast/$file.psiblast -out_ascii_pssm ../pssm/$file.pssm -num_threads 8
done
