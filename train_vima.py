import os
print("SSS", os.getcwd())
import torch
import argparse
import itertools
import pickle
import numpy as np
from unet_vima import Unet
from tqdm import tqdm
import random
import torch.optim as optim
from diffusion import GaussianDiffusion
from torchvision.utils import save_image
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding
from Scheduler import GradualWarmupScheduler
from vima_load import create_policy_from_ckpt
#from dataloader_cifar import load_data, transback
from dataloader_vima import load_data, transback
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from prompt_embedding import prepare_prompt

def train(params:argparse.Namespace):
    assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0 , 'please re-set your genbatch!!!'
    # initialize settings
    init_process_group(backend="nccl")
    # get local rank for each process
    local_rank = get_rank()
    # set device
    device = torch.device("cuda", local_rank)
    # load data
    dataloader, sampler = load_data(params.batchsize, params.numworkers)
    # initialize models
    net = Unet(
                in_ch = params.inch,
                mod_ch = params.modch,
                out_ch = params.outch,
                ch_mul = params.chmul,
                num_res_blocks = params.numres,
                cdim = params.cdim,
                use_conv = params.useconv,
                droprate = params.droprate,
                dtype = params.dtype
            )
    #cemblayer = ConditionalEmbedding(10, params.cdim, params.cdim).to(device)
    # load last epoch
    lastpath = os.path.join(params.moddir,'last_epoch.pt')
    '''
    if os.path.exists(lastpath):
        lastepc = torch.load(lastpath)['last_epoch']
        # load checkpoints
        checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_{lastepc}_checkpoint.pt'), map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        #cemblayer.load_state_dict(checkpoint['cemblayer'])
    else:
        lastepc = 0
    '''
    lastepc = 0
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    diffusion = GaussianDiffusion(
                    dtype = params.dtype,
                    model = net,
                    betas = betas,
                    w = params.w,
                    v = params.v,
                    device = device
                )
    
    # DDP settings 
    diffusion.model = DDP(
                            diffusion.model,
                            device_ids = [local_rank],
                            output_device = local_rank
                        )
    cemblayer = create_policy_from_ckpt("/home/rl/COT-diffusion/4M.ckpt", "cuda")
    cemblayer.to('cuda')
    #print("ssss",next(cemblayer.parameters()).device)
    # optimizer settings
    optimizer = torch.optim.AdamW(
                    itertools.chain(
                        diffusion.model.parameters()
                    ),
                    lr = params.lr,
                    weight_decay = 1e-4
                )
    
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = params.epoch,
                            eta_min = 0,
                            last_epoch = -1
                        )
    warmUpScheduler = GradualWarmupScheduler(
                            optimizer = optimizer,
                            multiplier = params.multiplier,
                            warm_epoch = params.epoch // 10,
                            after_scheduler = cosineScheduler,
                            last_epoch = lastepc
                        )
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler'])
    # training
    cnt = torch.cuda.device_count()
    for epc in range(lastepc, params.epoch):
        # turn into train mode
        diffusion.model.train()
        #cemblayer.train()
        sampler.set_epoch(epc)
        # batch iterations
        labs = []
        guids = []
        lab_seq = {}
        ids = 0
        with tqdm(dataloader, dynamic_ncols=True, disable=(local_rank % cnt != 0)) as tqdmDataLoader:

            #for img_0, img_1, lab, lab_1 in tqdmDataLoader:
            for img_0, img_1, pathes in tqdmDataLoader:
                optimizer.zero_grad()
                print("img_0", img_0, img_0.shape)
                print("img_1", img_1, img_1.shape)
                print("pathes", pathes, pathes.shape)
                # exit()

                id = random.randint(0, 1)
                ids =id
                if id == 0 :
                    x_0 = img_0.to(device)
                    x_1 = img_1.to(device)
                else:
                    x_0 = img_1.to(device)
                    x_1 = img_0.to(device)
                cemb_ = torch.zeros([params.batchsize, params.cdim], dtype=torch.float)
                print("params.batchsize", params.batchsize)
                for la in range(params.batchsize):
                    with open(pathes[la], "rb") as f:
                        traj_meta = pickle.load(f)

                    lab = traj_meta.pop("prompt")
                    lab_1 = traj_meta.pop("prompt_assets")

                    lab_ = lab#.to(device)
                    lab_1 = lab_1#.to(device)

                    print("prompt", lab_)
                    print("prompt_assets", lab_1)

                    # todo 这里的la没有意义？
                    prompt_token_type, word_batch, image_batch = prepare_prompt(lab_, lab_1, la)

                    word_batch = word_batch.to("cuda")
                    print("prompt_token_type", prompt_token_type, la)
                    image_batch = image_batch.to_torch_tensor(device="cuda")
                    print("word_batch", word_batch)
                    print("image_batch", image_batch)

                    #prompt_token_type = prompt_token_type.to("cuda")
                    # todo 这里的prompt_masks能不能用起来，VIMApolicy里是把o,a,p的token和mask token都放到policy nn里预测action
                    prompt_tokens, prompt_masks = cemblayer.forward_prompt_assembly(
                        (prompt_token_type, word_batch, image_batch)
                    )

                    # todo 额外加一个obj token


                    # todo 统一用vima policy  -->  作为diffusion的conditional input


                    #cemb,_ = cemblayer(lab)
                    #cemb[np.where(np.random.rand(b)<params.threshold)] = 0
                    prompt_tokens= prompt_tokens.squeeze(1)
                    print("prompt_tokens", prompt_tokens.size())
                    # exit()

                    # todo
                    # tokens_out = self.xattn_gpt(
                    #     obs_action_tokens=tokens,
                    #     prompt_tokens=prompt_token,
                    #     prompt_mask=prompt_token_mask,
                    #     obs_action_masks=masks.transpose(0, 1),
                    #     obs_action_position_ids=position_ids.transpose(0, 1),
                    #     prompt_position_ids=prompt_position_ids,
                    # )

                    cemb = torch.mean(prompt_tokens, dim=0)# change segmentation here       (H, 256)  12 10 14


                    print("cemb", cemb.size())   #(256)  (H)
                    exit()

                    #cemb_seq.append(cemb)
                    if la <= 4:
                        labs.append(cemb)
                        #print("SSs", x_1.size())
                        guids.append(x_1[la])


                    #lab_seq[lab_] = cemb

                    cemb_[la] = cemb       # (B, H)     (B, 256)
                exit()
                #cemb_seq = torch.cemb_seq
                #print("sss", cemb_.size())
                loss = diffusion.trainloss(x_0, x_1, num = id, cemb = cemb_)
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epc + 1,
                        "loss: ": loss.item(),
                        "batch per device: ":x_0.shape[0],
                        "img shape: ": x_0.shape[1:],
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )
        warmUpScheduler.step()
        # evaluation and save checkpoint
        if (epc + 1) % params.interval == 0:
            diffusion.model.eval()
            #cemblayer.eval()
            # generating samples
            # The model generate 40 pictures(8 per row) each time
            # pictures of same row belong to the same class
            all_samples = []
            each_device_batch = params.genbatch // cnt
            with torch.no_grad():
                #lab_list = labs[0:5]
                labnum = 5
                genshape = (each_device_batch, 3, 32, 64)
                dim_1 = each_device_batch // labnum
                cemb = torch.zeros([labnum,256],dtype=torch.float)
                gs = torch.zeros([labnum, 3,32,64], dtype=torch.float)
                for i in range(labnum):
                    cemb[i] = labs[i]
                    gs[i] = guids[i]
                cemb = cemb.unsqueeze(1).expand(-1, dim_1, -1).reshape(params.genbatch,-1)
                gs = gs.unsqueeze(1).expand(-1, dim_1, -1, -1, -1).reshape(params.genbatch,3,32,64)
                guid_num = torch.zeros(params.genbatch, 2).to("cuda")
                guid_num[:, ids] = 1.
            #for i in range(5):

                if params.ddim:

                    generated = diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select,  gs=gs, cemb = cemb , guid_num=guid_num)
                else:
                    generated = diffusion.sample(genshape , cemb = cemb, x_1=gs, guid_num=guid_num)
                img = transback(generated)
                #img = img.reshape( 3, 256, 128).contiguous()
                img = img.reshape(labnum, each_device_batch // labnum, 3, 32, 64).contiguous()
                gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
                all_gather(gathered_samples, img)
                all_samples.extend([img for img in gathered_samples])
                samples = torch.concat(all_samples, dim = 1).reshape(params.genbatch, 3, 32, 64)

                if local_rank == 0:
                    save_image(samples, os.path.join(params.samdir, f'generated_{epc+1}_pict.png'), nrow = params.genbatch)
            # save checkpoints
            checkpoint = {
                                'net':diffusion.model.module.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'scheduler':warmUpScheduler.state_dict()
                            }
            torch.save({'last_epoch':epc+1}, os.path.join(params.moddir,'last_epoch.pt'))
            torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc+1}_checkpoint.pt'))
        fw = open('condition.txt', 'wb')
        #pickle.dump(lab_seq, fw)
        torch.cuda.empty_cache()
    destroy_process_group()

def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize',type=int,default=128,help='batch size per device for training Unet model')
    parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim',type=int,default=256,help='dimension of conditional embedding')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0.1,help='dropout rate for model')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--lr',type=float,default=2e-4,help='learning rate')
    parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch',type=int,default=1500,help='epochs for training')
    parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
    parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
    parser.add_argument('--interval',type=int,default=20,help='epoch interval between two evaluations')
    parser.add_argument('--moddir',type=str,default='model',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--genbatch',type=int,default=40,help='batch size for sampling process')
    parser.add_argument('--clsnum',type=int,default=10,help='num of label classes')
    parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
