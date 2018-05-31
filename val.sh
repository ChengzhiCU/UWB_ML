enc_type='mlp'
checkpoint='/home/maocz/Project/models/03-29-21-38_mlp/best_model_epoch26_0.0356.zip'


python train.py --enc_type $enc_type --checkpoint $checkpoint
