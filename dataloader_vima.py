from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from data import myImageDataset as datasets
from torch.utils.data.distributed import DistributedSampler

def load_data(batchsize:int, numworkers:int):# -> tuple[DataLoader, DistributedSampler]:
    trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = datasets(
                        # img_dir = '/home/rl/COT-diffusion/rearrange',
                        # img_dir = '/home/rl/COT-diffusion/simple_manipulation',
                        # img_dir = '/home/rl/COT-diffusion/rearrange_then_restore',
                        # img_dir = '/home/rl/COT-diffusion/pick_in_order_then_restore',
                        img_dir = '/data/nifei/vima_v6/pick_in_order_then_restore',
                        #train = True,
                        #download = False,
                        transform = trans,
                    )
    sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        sampler = sampler,
                        drop_last = True
                    )
    return trainloader, sampler

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5
