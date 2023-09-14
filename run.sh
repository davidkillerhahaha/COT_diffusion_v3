export MASTER_ADDR=localhost
export MASTER_PORT=5678

#CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu train_vima.py
#CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu train_vima_v2.py
#CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_vima_v2.py
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_vima_v2.py
#CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu test_model.py

#CUDA_VISIBLE_DEVICES=1 python3 -u train_vima.py

#CUDA_VISIBLE_DEVICES=0 python -u pretrained_vae_encode.py


