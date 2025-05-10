import torch
from AIRP_read_data import read_dataset
from torch_geometric.loader import DataLoader

def read_dataloader(args):

    batch_size = args.batch_size

    train_list, val_list, test_list = read_dataset(
        args.data_path,
        max_atom_number=args.max_atom_number, max_atom_id = args.max_atom_id,
        train=args.train_ratio, val=args.val_ratio)

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size)
    test_loader = DataLoader(test_list, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def perturb_coordinates(x0, noise_std=0.3):

    noise = torch.randn_like(x0) * noise_std
    xT = x0 + noise
    return xT

def sub_center(x):

    center_of_mass = x.mean(dim=-2, keepdim=True)

    result = x - center_of_mass

    return result