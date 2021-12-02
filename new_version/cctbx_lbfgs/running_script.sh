for i in 0.3  0.6  0.9  1.2  1.5; do
cd $i

for k in $(seq 0 9);do
echo " -- $i.$k --"

prefix=$k"_minimized"

phenix.python /home/yanting/work/aniserver_1_qr/ani_qr_test/cctbx_lbfgs/qr-ani-test/run.py $k.pdb restraints=cctbx minimizer=lbfgs stpmax=0.2 max_itarations=2500 prefix=$prefix gradient_only=true > $k.log 2>&1

done

cd ..
done
