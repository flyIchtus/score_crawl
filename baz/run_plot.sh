for num in 1 2 3 4 6 11
do
	python3 plot.py -s=$num -n=0 &
	python3 plot.py -s=$num -n=1 &
done
for mm in 1 2 3 4 6 11
do
	python3 plot.py -s=$mm -m 0 1 &
done


