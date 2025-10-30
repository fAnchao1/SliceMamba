import torch
from engine import test_one_epoch
from datasets.dataset import Poly_datasets
import tqdm
from torch.utils.data import DataLoader
from configs.config_setting import setting_config
assert torch.cuda.is_available()

train_dataprovider, val_dataprovider = None, None

config = setting_config

print('#----------Preparing dataset----------#')

search_dataset = Poly_datasets(config.data_path, config, type='search')
search_dataprovider = DataLoader(search_dataset,
                            batch_size=32,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=config.num_workers,
                            drop_last=True)


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_miou_dice(model, cand):
    global search_dataprovider

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    miou = 0
    dice = 0

    print('starting test....')
    
    miou,dice = test_one_epoch(test_loader = search_dataprovider,
                    model = model,
                    architecture = cand,
                    config = config
                    )
   

    print('miou: {:.2f} dice: {:.2f}'.format(miou * 100, dice * 100))

    return miou, dice


def main():
    pass
