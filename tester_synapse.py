import torch
from engine_synapse import test_one_epoch
from datasets.dataset import Synapse_dataset
import tqdm
from torch.utils.data import DataLoader
from configs.config_setting_synapse import setting_config
assert torch.cuda.is_available()
from torchvision import transforms
from datasets.dataset import RandomGenerator
train_dataprovider, val_dataprovider = None, None

config = setting_config

print('#----------Preparing dataset----------#')
search_dataset = config.datasets(base_dir=config.search_path, split="search", list_dir=config.list_dir,
                                transform=transforms.Compose(
                                [RandomGenerator(output_size=[config.input_size_h, config.input_size_w])]))
search_sampler = DistributedSampler(search_dataset, shuffle=True) if config.distributed else None
search_dataprovider = DataLoader(search_dataset,
                            batch_size=32, # if config.distributed else config.batch_size,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=config.num_workers, 
                            sampler=search_sampler)



def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_dice_HD(model, cand):
    global search_dataprovider

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dice = 0
    HD95 = 0

    print('starting test....')
    
    dice = test_one_epoch(test_datasets = search_dataset,
                    test_loader = search_dataprovider,
                    model = model,
                    architecture = cand,
                    config = config,
                    test_save_path = None,
                    val_or_test = False
                    )
   

    print('dice: {:.2f} '.format(dice * 100))

    return dice


def main():
    pass
