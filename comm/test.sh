# CUB conv4
python ./test.py --dataset CUB --model Conv4 --method baseline                    --train_aug --gpu 0 
python ./test.py --dataset CUB --model Conv4 --method maml             --n_shot 5 --train_aug --gpu 0 
python ./test.py --dataset CUB --model Conv4 --method matchingnet      --n_shot 5 --train_aug --gpu 0 
python ./test.py --dataset CUB --model Conv4 --method protonet         --n_shot 5 --train_aug --gpu 0
python ./test.py --dataset CUB --model Conv4 --method relationnet      --n_shot 5 --train_aug --gpu 0 
python ./test.py --dataset CUB --model Conv4 --method relationnet_ours --n_shot 5 --train_aug --gpu 0