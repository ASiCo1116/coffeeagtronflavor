for (( i = 1; i <= ${1}; i++ )); do
	python3 carspls.py -pc ${i} -rn 50 -ra 0.9 -ps 3 -fn 0521cars_agtron_sg_msc
done