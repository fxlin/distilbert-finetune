#for step in 50 100 200 300 400 500
#do
#	for fix_t in 0.1 0.2 0.3 0.4 0.5 0.6
#	do
#	  python3 distillbert-1adp2fixed.py $step $fix_t
#	done
#done

#for step in 50 100 200 300 400 500
#do
#	  python3 distillbert-1adp2fixed.py $step 0
#done

for stage1_step in 50 100 200 300 400 500
do
	for stage0_step in 50 100 200 300 400 500
	do
	  python3 distillbert-0find1adp2fixed.py $stage1_step $stage0_step
	done
done