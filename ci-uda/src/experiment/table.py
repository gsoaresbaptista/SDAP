import json
from pathlib import Path

import torch

from experiment.manager import setup_dataloader
from models.base_model import PrototypeResNet
from training.train_pldca import PLDCA
from training.train_proca import ProCA
from utils.settings import set_seed

if __name__ == '__main__':
    # data_dir = Path('datasets/office-home')
    data_dir = Path('datasets/office-31')
    # config_dir = Path('data/office_home/config.json')
    config_dir = Path('data/office_31/config.json')

    num_classes = 65
    learning_rate = 0.001
    batch_size = 24
    METHOD = 'proca'
    num_epochs = 1

    proca_hiperparams = {'threshold': 0.15}
    pldca_hiperparams = {'threshold': 0.10, 'beta': 0.6, 'warm_up': 6, 'step_update': 3}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # DOMAINS = ['art', 'clipart', 'product', 'real_world'] # office-home
    DOMAINS = ['Amazon', 'DSLR', 'Webcam'] # office-31
    results = {}

    print(METHOD)
    for source in DOMAINS:
        for target in DOMAINS:
            if source == target:
                continue
            print(source, target)

            pretrained_path = Path(f'weights/{source}_source.pth')
            data_path = data_dir / target

            pretrained_model = PrototypeResNet(num_classes, pretrained_path).to(device)
            s_loader = setup_dataloader(data_dir / source, config_dir, range(65), batch_size, False)
            set_seed(42)

            method_params = proca_hiperparams if METHOD == 'proca' else pldca_hiperparams
            method_class = ProCA if METHOD == 'proca' else PLDCA

            proca = method_class(
                pretrained_model,
                s_loader,
                learning_rate,
                device=device,
                **method_params,
            )

            for i in range(num_classes // 10):
                train_ids = range(i * 10, (i + 1) * 10)
                val_ids = range(0, (i + 1) * 10)
                t_loader = setup_dataloader(
                    data_path,
                    config_dir,
                    train_ids,
                    batch_size,
                    False,
                    True,
                    True,
                    METHOD == 'pldca',
                )
                val_loader = setup_dataloader(
                    data_path, config_dir, val_ids, batch_size, False, False
                )
                proca.increment(t_loader, val_loader, num_epochs)

            results[f'{source}_{target}'] = proca.last_acc.item()
            with open(f'{METHOD}_table_barlow_twins.json', 'w') as json_file:
                json.dump(results, json_file)
