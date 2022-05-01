#for step in 50 100 200 300 400 500
#do
#	  python3 distillbert-1adp2adp.py $step
#done

for step in 50 100 200 300 400 500
do
	for fix_t in 0.4 0.45 0.5 0.55 0.6 0.65
	do
	  python3 distillbert-0find1fixed2fixed.py $step $fix_t
	done
done