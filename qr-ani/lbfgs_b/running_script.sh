for i in 0.3  0.6  0.9  1.2  1.5; do
cd $i

for k in $(seq 0 9);do
echo " -- $i.$k --"

qr.refine $k.pdb  mode=opt restraints=qm engine_name=torchani clustering=false number_of_micro_cycles=100 max_iterations_refine=50 minimizer=lbfgsb > $k.log 2>&1

cp  pdb/*_refined.pdb .

fold_name=$k"_pdb"
mv pdb/ $fold_name/

done

cd ..
done
