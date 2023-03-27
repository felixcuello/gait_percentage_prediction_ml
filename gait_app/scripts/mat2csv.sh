for i in `ls /data/*mat`; do python3 /app/scripts/mat2csv.py $i $i.csv; done
