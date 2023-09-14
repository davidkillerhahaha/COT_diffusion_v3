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
from diffusion_v2 import GaussianDiffusion_v2
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



def test(params: argparse.Namespace):
    assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0, 'please re-set your genbatch!!!'
    # initialize settings
    init_process_group(backend="nccl")
    # get local rank for each process
    local_rank = get_rank()
    # set device
    device = torch.device("cuda", local_rank)
    # load data
    dataloader, sampler = load_data(params.batchsize, params.numworkers)


    # if params.log_to_wandb:
    #     test = wandb.init(
    #         project='COT-Diffusion',
    #         # config=variant
    #     )
    #     test.config.update(dict(epoch=params.epoch, lr=params.lr, batch_size=params.batchsize))


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

    checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_46_checkpoint.pt'))
    # print('checkpoint: ', checkpoint['net'])

    # diffusion.model.load_state_dict(torch.load(os.path.join(params.moddir, f'ckpt_46_checkpoint.pt'))['net'])
    diffusion.model.load_state_dict(torch.load(os.path.join(params.moddir, f'ckpt_740_checkpoint.pt'))['net'])

    diffusion.model.eval()
    cemblayer.eval()
    cnt = torch.cuda.device_count()

    # # test
    # generation_index = 0
    # generated_image = np.load(os.path.join(params.samdir,
    #                                       f'sample_{params.epoch}_generation_index_{generation_index}_v1.npz'), allow_pickle=True)
    # print('generated_image:', generated_image, generated_image.files)
    #
    # generated_image = (generated_image.cpu() * 255).numpy().clamp(0, 255)
    #
    # np.savez(os.path.join(params.samdir,
    #                       f'sample_{params.epoch}_generation_index_{generation_index}_v1.npz'), generated_image)
    #
    # save_image(generated_image,
    #            os.path.join(params.samdir, f'sample_{params.epoch}_generation_index_{generation_index}_v2.png'),
    #            nrow=params.batchsize)
    # exit()



    with torch.no_grad():
        with tqdm(dataloader, dynamic_ncols=True, disable=(local_rank % cnt != 0)) as tqdmDataLoader:
            for unnorm_image_dict, image_dict, traj_path, images_num in tqdmDataLoader:
                unnorm_initial_image = unnorm_image_dict['0'].to('cuda')
                initial_image_token = cemblayer.obj_encoder.cropped_img_encoder(unnorm_initial_image).to('cuda')
                # print('initial_image_token:', initial_image_token.shape)

                cemb_list = torch.zeros([params.batchsize, params.vanilla_cdim], dtype=torch.float).to('cuda')

                for batch_id in range(params.batchsize):
                    with open(traj_path[batch_id], "rb") as f:
                        try:
                            traj_meta = pickle.load(f)
                        except EOFError:
                            print("EOF Error! turn to next batch!")
                            continue

                    # multi modal prompt tokens
                    prompt = traj_meta.pop("prompt")
                    print('id: ', batch_id, '  prompt: ', prompt)
                    prompt_assets = traj_meta.pop("prompt_assets")
                    prompt_token_type, word_batch, image_batch = prepare_prompt(prompt=prompt,
                                                                                prompt_assets=prompt_assets)
                    word_batch = word_batch.to("cuda")
                    # print("prompt_token_type", prompt_token_type)
                    image_batch = image_batch.to_torch_tensor(device="cuda")

                    prompt_tokens, prompt_masks = cemblayer.forward_prompt_assembly(
                        (prompt_token_type, word_batch, image_batch)
                    )

                    prompt_tokens = prompt_tokens.squeeze(1)
                    # print("prompt_tokens", prompt_tokens.size())

                    cemb = torch.mean(prompt_tokens, dim=0)  # change segmentation here       (H, 256)  12 10 14
                    # print("cemb", cemb.size())  # (256)  (H)
                    cemb_list[batch_id] = cemb.to('cuda')  # (B, H)     (B, 256)

                condition_info = torch.concat([initial_image_token, cemb_list], dim=1).to('cuda')

                genshape = (params.batchsize, 3, 32, 64)

                images_num = 3
                for generation_index in range(images_num):
                    generation_index = 1
                    print('start generation {} step '.format(generation_index))
                    if generation_index == 0:
                        initial_image = image_dict['0'].to('cuda')
                        second_image = image_dict['1'].to('cuda')
                        generated_image = diffusion.sample(genshape, cemb=condition_info, image_dict=image_dict, conditioned_image= initial_image).to('cuda')
                    else:
                        initial_image = image_dict['0'].to('cuda')
                        second_image = image_dict['1'].to('cuda')
                        # conditioned_image_is_generated = True
                        conditioned_image_is_generated = False
                        if conditioned_image_is_generated == True:
                            generated_image = diffusion.sample(genshape, cemb=condition_info, image_dict=image_dict,
                                                         conditioned_image=generated_image.to('cuda'))
                        else:
                            gt_image = image_dict['1'].to('cuda')
                            generated_image = diffusion.sample(genshape, cemb=condition_info, image_dict=image_dict,
                                                               conditioned_image=gt_image)
                    print('generated_image:', generated_image, generated_image.shape)

                    # transform samples into images
                    generated_image = transback(generated_image)
                    initial_image = transback(initial_image)
                    second_image = transback(second_image)
                    gt_image = transback(gt_image)
                    save_image(generated_image, os.path.join(params.samdir, f'sample_{params.epoch}_generation_index_{generation_index}_generated.png'),
                               nrow=params.batchsize)

                    save_image(second_image, os.path.join(params.samdir, f'sample_{params.epoch}_generation_index_{generation_index}_second.png'),
                               nrow=params.batchsize)

                    save_image(initial_image, os.path.join(params.samdir, f'sample_{params.epoch}_generation_index_{generation_index}_initial.png'),
                               nrow=params.batchsize)

                    save_image(gt_image, os.path.join(params.samdir, f'sample_{params.epoch}_generation_index_{generation_index}_gt.png'),
                               nrow=params.batchsize)


                    # np.savez(os.path.join(params.samdir,
                    #                       f'sample_{params.epoch}_generation_index_{generation_index}_v1.npz'), generated_image)

                    generated_image = (generated_image.cpu() * 255).clamp(0, 255)


                    np.savez(os.path.join(params.samdir,
                                          f'sample_{params.epoch}_generation_index_{generation_index}_v1.npz'), generated_image)

                    save_image(generated_image, os.path.join(params.samdir, f'sample_{params.epoch}_generation_index_{generation_index}_v2.png'),
                               nrow=params.batchsize)

                    exit()


                exit()


                generated = diffusion.sample(image_dict=image_dict, images_num=int(images_num[batch_id]), conditioned_info=condition_info)


                break

#             # transform samples into images
#         img = transback(generated)
#         img = img.reshape(params.clsnum, each_device_batch // params.clsnum, 3, 32, 32).contiguous()
#         gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
#         all_gather(gathered_samples, img)
#         all_samples.extend([img.cpu() for img in gathered_samples])
#
#
# samples = torch.concat(all_samples, dim=1).reshape(params.genbatch * numloop, 3, 32, 32)
# if local_rank == 0:
#     print(samples.shape)
#     # save images
#     if params.fid:
#         samples = (samples * 255).clamp(0, 255).to(torch.uint8)
#         samples = samples.permute(0, 2, 3, 1).numpy()[:params.genum]
#         print(samples.shape)
#         np.savez(os.path.join(params.samdir, f'sample_{samples.shape[0]}_diffusion_{params.epoch}_{params.w}.npz'),
#                  samples)
#     else:
#         save_image(samples, os.path.join(params.samdir, f'sample_{params.epoch}_pict_{params.w}.png'),
#                    nrow=params.genbatch // params.clsnum)
# destroy_process_group()








def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize', type=int, default=16, help='batch size per device for training Unet model')
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
    parser.add_argument('--interval', type=int, default=1, help='epoch interval between two evaluations')
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
    test(args)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')  # good solution !!!!
    import torch, gc

    gc.collect()
    torch.cuda.empty_cache()
    main()
