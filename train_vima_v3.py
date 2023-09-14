import os

print("SSS", os.getcwd())
import torch
import argparse
import itertools
import pickle
import numpy as np
# from unet_vima import Unet
from unet_vima_v2 import Unet
from tqdm import tqdm
import random
import torch.optim as optim
# from diffusion import GaussianDiffusion
from diffusion_v3 import GaussianDiffusion_v2
# from diffusion_v3 import GaussianDiffusion_v3
from torchvision.utils import save_image
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding
from Scheduler import GradualWarmupScheduler
from vima_load import create_policy_from_ckpt
# from dataloader_cifar import load_data, transback
from dataloader_vima import load_data, transback
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from prompt_embedding import prepare_prompt
from diffusers.models import AutoencoderKL
import wandb

########################################################################################################################
## todo     for debug
# from __future__ import annotations

# import os
#
# import numpy as np
# from tokenizers import Tokenizer
# from tokenizers import AddedToken
# from einops import rearrange
# import cv2
# from vima.utils import *
# from vima import create_policy_from_ckpt
# from vima_bench import *
# from gym.wrappers import TimeLimit as _TimeLimit
# from gym import Wrapper
# import torch
# import argparse
#
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
#
#
# _kwargs = {
#     "single_word": True,
#     "lstrip": False,
#     "rstrip": False,
#     "normalized": True,
# }
#
# PLACEHOLDER_TOKENS = [
#     AddedToken("{base_obj}", **_kwargs),
#     AddedToken("{base_obj_1}", **_kwargs),
#     AddedToken("{base_obj_2}", **_kwargs),
#     AddedToken("{dragged_obj}", **_kwargs),
#     AddedToken("{dragged_obj_1}", **_kwargs),
#     AddedToken("{dragged_obj_2}", **_kwargs),
#     AddedToken("{dragged_obj_3}", **_kwargs),
#     AddedToken("{dragged_obj_4}", **_kwargs),
#     AddedToken("{dragged_obj_5}", **_kwargs),
#     AddedToken("{swept_obj}", **_kwargs),
#     AddedToken("{bounds}", **_kwargs),
#     AddedToken("{constraint}", **_kwargs),
#     AddedToken("{scene}", **_kwargs),
#     AddedToken("{demo_blicker_obj_1}", **_kwargs),
#     AddedToken("{demo_less_blicker_obj_1}", **_kwargs),
#     AddedToken("{demo_blicker_obj_2}", **_kwargs),
#     AddedToken("{demo_less_blicker_obj_2}", **_kwargs),
#     AddedToken("{demo_blicker_obj_3}", **_kwargs),
#     AddedToken("{demo_less_blicker_obj_3}", **_kwargs),
#     AddedToken("{start_scene}", **_kwargs),
#     AddedToken("{end_scene}", **_kwargs),
#     AddedToken("{before_twist_1}", **_kwargs),
#     AddedToken("{after_twist_1}", **_kwargs),
#     AddedToken("{before_twist_2}", **_kwargs),
#     AddedToken("{after_twist_2}", **_kwargs),
#     AddedToken("{before_twist_3}", **_kwargs),
#     AddedToken("{after_twist_3}", **_kwargs),
#     AddedToken("{frame_0}", **_kwargs),
#     AddedToken("{frame_1}", **_kwargs),
#     AddedToken("{frame_2}", **_kwargs),
#     AddedToken("{frame_3}", **_kwargs),
#     AddedToken("{frame_4}", **_kwargs),
#     AddedToken("{frame_5}", **_kwargs),
#     AddedToken("{frame_6}", **_kwargs),
#     AddedToken("{ring}", **_kwargs),
#     AddedToken("{hanoi_stand}", **_kwargs),
#     AddedToken("{start_scene_1}", **_kwargs),
#     AddedToken("{end_scene_1}", **_kwargs),
#     AddedToken("{start_scene_2}", **_kwargs),
#     AddedToken("{end_scene_2}", **_kwargs),
#     AddedToken("{start_scene_3}", **_kwargs),
#     AddedToken("{end_scene_3}", **_kwargs),
# ]
# PLACEHOLDERS = [token.content for token in PLACEHOLDER_TOKENS]
# tokenizer = Tokenizer.from_pretrained("t5-base")
# tokenizer.add_tokens(PLACEHOLDER_TOKENS)

########################################################################################################################


def train(params: argparse.Namespace):
    assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0, 'please re-set your genbatch!!!'
    # initialize settings
    init_process_group(backend="nccl")
    # get local rank for each process
    local_rank = get_rank()
    # set device
    device = torch.device("cuda", local_rank)
    # load data
    dataloader, sampler = load_data(params.batchsize, params.numworkers)


    # todo
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to('cuda')

    if params.log_to_wandb:
        test = wandb.init(
            project='COT-Diffusion', entity='cot-diffusion'
            # config=variant
        )
        test.config.update(dict(epoch=params.epoch, lr=params.lr, batch_size=params.batchsize))


        # if not os.path.exists(save_path): os.mkdir(save_path)


    # initialize models
    net = Unet(
        in_ch=params.inch,
        mod_ch=params.modch,
        out_ch=params.outch,
        ch_mul=params.chmul,
        num_res_blocks=params.numres,
        cdim=params.cdim,
        use_conv=params.useconv,
        droprate=params.droprate,
        dtype=params.dtype
    )


    lastepc = 0
    betas = get_named_beta_schedule(num_diffusion_timesteps=params.T)
    diffusion = GaussianDiffusion_v2(
        dtype=params.dtype,
        model=net,
        betas=betas,
        w=params.w,
        v=params.v,
        device=device
    )

    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # exit()

    # # DDP settings
    # diffusion.model = DDP(
    #     diffusion.model,
    #     device_ids=[local_rank],
    #     output_device=local_rank
    # )




    cemblayer = create_policy_from_ckpt("/home/rl/COT-diffusion/4M.ckpt", "cuda")
    cemblayer.to('cuda')
    # print("ssss",next(cemblayer.parameters()).device)
    # optimizer settings
    optimizer = torch.optim.AdamW(
        itertools.chain(
            diffusion.model.parameters()
        ),
        lr=params.lr,
        weight_decay=1e-4
    )

    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=params.epoch,
        eta_min=0,
        last_epoch=-1
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=params.multiplier,
        warm_epoch=params.epoch // 10,
        after_scheduler=cosineScheduler,
        last_epoch=lastepc
    )
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler'])




########################################################################################################################
    # for debuging

    # class ResetFaultToleranceWrapper(Wrapper):
    #     max_retries = 10
    #
    #     def __init__(self, env):
    #         super().__init__(env)
    #
    #     def reset(self):
    #         for _ in range(self.max_retries):
    #             try:
    #                 return self.env.reset()
    #             except:
    #                 current_seed = self.env.unwrapped.task.seed
    #                 self.env.global_seed = current_seed + 1
    #         raise RuntimeError(
    #             "Failed to reset environment after {} retries".format(self.max_retries)
    #         )
    #
    # class TimeLimitWrapper(_TimeLimit):
    #     def __init__(self, env, bonus_steps: int = 0):
    #         super().__init__(env, env.task.oracle_max_steps + bonus_steps)
    #
    #
    # # policy = create_policy_from_ckpt(cfg.ckpt, cfg.device)
    # env = TimeLimitWrapper(
    #     ResetFaultToleranceWrapper(
    #         make(
    #             cfg.task,
    #             modalities=["segm", "rgb"],
    #             task_kwargs=PARTITION_TO_SPECS["test"][cfg.partition][cfg.task],
    #             seed=seed,
    #             render_prompt=True,
    #             display_debug_window=True,
    #             hide_arm_rgb=False,
    #         )
    #     ),
    #     bonus_steps=2,
    # )
    #
    # env.global_seed = seed
    #
    # obs = env.reset()
    # print('obs:', obs)
    # exit()
########################################################################################################################

    # training
    cnt = torch.cuda.device_count()
    for epc in range(lastepc, params.epoch):
        # turn into train mode
        diffusion.model.train()
        # cemblayer.train()
        sampler.set_epoch(epc)
        # batch iterations
        with tqdm(dataloader, dynamic_ncols=True, disable=(local_rank % cnt != 0)) as tqdmDataLoader:
            for unnorm_image_dict, image_dict, traj_path, images_num in tqdmDataLoader:
                optimizer.zero_grad()
                unnorm_initial_image = unnorm_image_dict['0'].to('cuda')
                # print("unnorm_initial_image", unnorm_initial_image, unnorm_initial_image.shape)

                # A. huggingface vit -> token
                # B. cropped_img_encoder tokenizer
                # visual token via

                initial_image_token = cemblayer.obj_encoder.cropped_img_encoder(unnorm_initial_image).to('cuda')
                # print('initial_image_token:', initial_image_token.shape)

                cemb_list = torch.zeros([params.batchsize, params.vanilla_cdim], dtype=torch.float).to('cuda')

                for batch_id in range(params.batchsize):
                    with open(traj_path[batch_id], "rb") as f:
                        # print('batch_id:', batch_id)
                        try:
                            traj_meta = pickle.load(f)
                        except EOFError:
                            print("EOF Error! turn to next batch!")
                            continue

                    # multi modal prompt tokens
                    prompt = traj_meta.pop("prompt")
                    prompt_assets = traj_meta.pop("prompt_assets")
                    prompt_token_type, word_batch, image_batch = prepare_prompt(prompt=prompt, prompt_assets=prompt_assets)

                    word_batch = word_batch.to("cuda")
                    # print("prompt_token_type", prompt_token_type)
                    image_batch = image_batch.to_torch_tensor(device="cuda")
                    # print("word_batch", word_batch, word_batch.shape)
                    # print("image_batch", image_batch, image_batch.shape)
                    # prompt_token_type = prompt_token_type.to("cuda")

                    prompt_tokens, prompt_masks = cemblayer.forward_prompt_assembly(
                        (prompt_token_type, word_batch, image_batch)
                    )

                    prompt_tokens = prompt_tokens.squeeze(1)
                    # print("prompt_tokens", prompt_tokens.size())

                    cemb = torch.mean(prompt_tokens, dim=0)  # change segmentation here       (H, 256)  12 10 14
                    # print("cemb", cemb.size())  # (256)  (H)
                    cemb_list[batch_id] = cemb.to('cuda')  # (B, H)     (B, 256)
                # print('cemb_list:', cemb_list.shape)
                # print('initial_image_token:', initial_image_token.shape)
                condition_info = torch.concat([initial_image_token, cemb_list], dim=1).to('cuda')
                # print('condition_info:', condition_info, condition_info.shape)

                loss, loss_dict = diffusion.trainloss_loop(image_dict=image_dict, images_num=int(images_num[batch_id]), conditioned_info=condition_info)
                # print('loss:', loss, loss.shape)
                # print('loss_dict:', loss_dict)

                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epc + 1,
                        "loss: ": loss.item(),
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )

                test.log({'epoch': epc, 'loss': loss.item(), 'loss_1':loss_dict['0'].item(), 'loss_2':loss_dict['1'].item()})

        warmUpScheduler.step()
        # evaluation and save checkpoint
        if (epc + 1) % params.interval == 0:

            # save checkpoints
            checkpoint = {
                'net': diffusion.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': warmUpScheduler.state_dict()
            }
            torch.save({'last_epoch': epc + 1}, os.path.join(params.moddir, 'last_epoch_v3.pt'))
            torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc + 1}_checkpoint_v3.pt'))


            # diffusion.model.eval()
            # # cemblayer.eval()
            # # generating samples
            # # The model generate 40 pictures(8 per row) each time
            # # pictures of same row belong to the same class
            # all_samples = []
            # each_device_batch = params.genbatch // cnt
            # with torch.no_grad():
            #
            #     unnorm_image_dict, image_dict, traj_path, images_num = next(iter(dataloader))
            #     with tqdm(dataloader, dynamic_ncols=True, disable=(local_rank % cnt != 0)) as tqdmDataLoader:
            #         for unnorm_image_dict, image_dict, traj_path, images_num in tqdmDataLoader:
            #             unnorm_initial_image = unnorm_image_dict['0'].to('cuda')
            #             initial_image_token = cemblayer.obj_encoder.cropped_img_encoder(unnorm_initial_image).to('cuda')
            #             # print('initial_image_token:', initial_image_token.shape)
            #
            #             cemb_list = torch.zeros([params.batchsize, params.vanilla_cdim], dtype=torch.float).to('cuda')
            #
            #             for batch_id in range(params.batchsize):
            #                 with open(traj_path[batch_id], "rb") as f:
            #                     try:
            #                         traj_meta = pickle.load(f)
            #                     except EOFError:
            #                         print("EOF Error! turn to next batch!")
            #                         continue
            #
            #                 # multi modal prompt tokens
            #                 prompt = traj_meta.pop("prompt")
            #                 prompt_assets = traj_meta.pop("prompt_assets")
            #                 prompt_token_type, word_batch, image_batch = prepare_prompt(prompt=prompt,
            #                                                                             prompt_assets=prompt_assets)
            #                 word_batch = word_batch.to("cuda")
            #                 # print("prompt_token_type", prompt_token_type)
            #                 image_batch = image_batch.to_torch_tensor(device="cuda")
            #
            #                 prompt_tokens, prompt_masks = cemblayer.forward_prompt_assembly(
            #                     (prompt_token_type, word_batch, image_batch)
            #                 )
            #
            #                 prompt_tokens = prompt_tokens.squeeze(1)
            #                 # print("prompt_tokens", prompt_tokens.size())
            #
            #                 cemb = torch.mean(prompt_tokens, dim=0)  # change segmentation here       (H, 256)  12 10 14
            #                 # print("cemb", cemb.size())  # (256)  (H)
            #                 cemb_list[batch_id] = cemb.to('cuda')  # (B, H)     (B, 256)
            #
            #             condition_info = torch.concat([initial_image_token, cemb_list], dim=1).to('cuda')
            #
            #             generated = diffusion.sample(image_dict=image_dict, images_num=int(images_num[batch_id]), conditioned_info=condition_info)
            #
            #
            #
            #     labnum = 5
            #     genshape = (each_device_batch, 3, 32, 64)
            #     dim_1 = each_device_batch // labnum
            #     cemb = torch.zeros([labnum, 256], dtype=torch.float)
            #     gs = torch.zeros([labnum, 3, 32, 64], dtype=torch.float)
            #     for i in range(labnum):
            #         cemb[i] = labs[i]
            #         gs[i] = guids[i]
            #     cemb = cemb.unsqueeze(1).expand(-1, dim_1, -1).reshape(params.genbatch, -1)
            #     gs = gs.unsqueeze(1).expand(-1, dim_1, -1, -1, -1).reshape(params.genbatch, 3, 32, 64)
            #     guid_num = torch.zeros(params.genbatch, 2).to("cuda")
            #     guid_num[:, ids] = 1.
            #     # for i in range(5):
            #
            #     if params.ddim:
            #
            #         generated = diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select, gs=gs,
            #                                           cemb=cemb, guid_num=guid_num)
            #     else:
            #         generated = diffusion.sample(genshape, cemb=cemb, x_1=gs, guid_num=guid_num)
            #     img = transback(generated)
            #     # img = img.reshape( 3, 256, 128).contiguous()
            #     img = img.reshape(labnum, each_device_batch // labnum, 3, 32, 64).contiguous()
            #     gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
            #     all_gather(gathered_samples, img)
            #     all_samples.extend([img for img in gathered_samples])
            #     samples = torch.concat(all_samples, dim=1).reshape(params.genbatch, 3, 32, 64)
            #
            #     if local_rank == 0:
            #         save_image(samples, os.path.join(params.samdir, f'generated_{epc + 1}_pict.png'),
            #                    nrow=params.genbatch)
            # # save checkpoints
            # checkpoint = {
            #     'net': diffusion.model.module.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'scheduler': warmUpScheduler.state_dict()
            # }
            # torch.save({'last_epoch': epc + 1}, os.path.join(params.moddir, 'last_epoch.pt'))
            # torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc + 1}_checkpoint.pt'))
        # pickle.dump(lab_seq, fw)
        torch.cuda.empty_cache()
    destroy_process_group()


def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize', type=int, default=24, help='batch size per device for training Unet model')
    parser.add_argument('--numworkers', type=int, default=1, help='num workers for training Unet model')
    parser.add_argument('--inch', type=int, default=3, help='input channels for Unet model')
    # parser.add_argument('--modch', type=int, default=64, help='model channels for Unet model')
    parser.add_argument('--modch', type=int, default=32, help='model channels for Unet model')
    parser.add_argument('--T', type=int, default=1000, help='timesteps for Unet model')
    parser.add_argument('--outch', type=int, default=3, help='output channels for Unet model')
    parser.add_argument('--chmul', type=list, default=[1, 2, 2, 2], help='architecture parameters training Unet model')
    parser.add_argument('--numres', type=int, default=2, help='number of resblocks for each block in Unet model')
    parser.add_argument('--vanilla_cdim', type=int, default=256, help='vanilla dimension of conditional embedding')
    parser.add_argument('--cdim', type=int, default=1024, help='dimension of conditional embedding')
    parser.add_argument('--useconv', type=bool, default=True, help='whether use convlution in downsample')
    parser.add_argument('--droprate', type=float, default=0.1, help='dropout rate for model')
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--w', type=float, default=1.8, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v', type=float, default=0.3,
                        help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch', type=int, default=10000, help='epochs for training')
    parser.add_argument('--multiplier', type=float, default=2.5, help='multiplier for warmup')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold for classifier-free guidance')
    parser.add_argument('--interval', type=int, default=20, help='epoch interval between two evaluations')
    parser.add_argument('--moddir', type=str, default='model', help='model addresses')
    parser.add_argument('--samdir', type=str, default='sample', help='sample addresses')
    parser.add_argument('--genbatch', type=int, default=40, help='batch size for sampling process')
    parser.add_argument('--clsnum', type=int, default=10, help='num of label classes')
    parser.add_argument('--num_steps', type=int, default=50, help='sampling steps for DDIM')
    parser.add_argument('--eta', type=float, default=0, help='eta for variance during DDIM sampling process')
    parser.add_argument('--select', type=str, default='linear', help='selection stragies for DDIM')
    parser.add_argument('--ddim', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False,
                        help='whether to use ddim')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')  # good solution !!!!
    import torch, gc

    gc.collect()
    torch.cuda.empty_cache()
    main()
