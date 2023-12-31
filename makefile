.PHONY : train
train:
	CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu train_vima.py
.PHONY : sample
samplepict:
	CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu sample.py --ddim True --select quadratic --genbatch 80 --w 0.5
.PHONY : samplenpz
samplenpz:
	CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu sample.py --fid True
.PHONY : clean
clean:
	rm -rf __pycache__
	rm -rf model/*
	rm -rf sample/*
