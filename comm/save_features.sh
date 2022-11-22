# CUB conv4
python ./save_features.py --dataset CUB --model Conv4 --method baseline                    --train_aug --gpu 0 
python ./save_features.py --dataset CUB --model Conv4 --method maml             --n_shot 5 --train_aug --gpu 0 
python ./save_features.py --dataset CUB --model Conv4 --method matchingnet      --n_shot 5 --train_aug --gpu 0 
python ./save_features.py --dataset CUB --model Conv4 --method protonet         --n_shot 5 --train_aug --gpu 0
python ./save_features.py --dataset CUB --model Conv4 --method relationnet      --n_shot 5 --train_aug --gpu 0 
python ./save_features.py --dataset CUB --model Conv4 --method relationnet_ours --n_shot 5 --train_aug --gpu 0 