d=mlc-runs/`date | sed 's| |_|g'`
log=$d/log.txt
mkdir -p $d

python train.py
tee $log