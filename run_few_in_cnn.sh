enc_type="cnn"
epochs='20'
marg_lambda='0.9'
lr='1e-2'
lambda_vae='0.3'
data_tr_per='0.075'
data_val_per='0.075'
batch='64'
seed='2333'
python3 train.py --seed $seed --enc_type $enc_type --epochs $epochs --marg_lambda $marg_lambda --lr $lr --lambda_vae $lambda_vae --little_input --data_tr_per $data_tr_per --data_val_per $data_val_per --batch $batch

