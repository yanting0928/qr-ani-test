for i in 0.3  0.6  0.9  1.2  1.5; do
cd $i

for k in $(seq 0 9);do
echo " -- $i.$k --"

phenix.python /home/yanting/work/aniserver_1_qr/ani_qr_test/cctbx_lbfgs/qr-ani-test/run.py $k.pdb restraints=cctbx minimizer=lbfgs_ase stpmax=0.2 max_itarations=50 macro_cycles=50  > $k.log 2>&1

done

cd ..
done
