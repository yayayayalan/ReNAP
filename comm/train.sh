# CUB conv4
nohup python ./train.py --dataset CUB --model Conv4 --method baseline                    --train_aug --gpu 0 > baseline_conv4_CUB.out 2>&1 &
nohup python ./train.py --dataset CUB --model Conv4 --method maml             --n_shot 5 --train_aug --gpu 0 > maml_conv4_CUB.out 2>&1 &
nohup python ./train.py --dataset CUB --model Conv4 --method matchingnet      --n_shot 5 --train_aug --gpu 0 > matchingnet_conv4_CUB.out 2>&1 &
nohup python ./train.py --dataset CUB --model Conv4 --method protonet         --n_shot 5 --train_aug --gpu 0 > protonet_conv4_CUB.out 2>&1 &
nohup python ./train.py --dataset CUB --model Conv4 --method relationnet      --n_shot 5 --train_aug --gpu 0 > relationnet_CUB_conv4.out 2>&1 &
nohup python ./train.py --dataset CUB --model Conv4 --method relationnet_ours --n_shot 5 --train_aug --gpu 0 > relationnet_ours_CUB_conv4.out 2>&1 &