export CUDA_VISIBLE_DEVICES=0,1
python3 main.py --dataset celebA --epoch 300 --batch_size 64 --checkpoint_dir checkpoint/BEGAN-CS --result_dir results/BEGAN-CS --log_dir logs/BEGAN-CS --g_lr 0.0001 --d_lr 0.0001 --train false
