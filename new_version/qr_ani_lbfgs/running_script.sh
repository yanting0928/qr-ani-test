for i in 1.5 1.2 0.9 0.6 0.3; do
cd $i

for k in $(seq 0 9);do
echo " -- $i.$k --"

qr.refine $k.pdb  mode=opt  restraints=qm engine_name=torchani clustering=false stpmax=0.2 max_iterations_refine=2500 minimizer=lbfgs number_of_micro_cycles=1 gradient_only=true > $k.log 2>&1

cp  pdb/*_refined.pdb .

fold_name=$k"_pdb"
mv pdb/ $fold_name/

done

cd ..
done
