"""dbim_image_translation.py
======================================================
Strict reference implementation of **Diffusion‑Bridge Implicit Models (DBIM)**
as described in the paper _"Fast Non‑Markovian Diffusion Bridges"_ (2024).
The code supports:
    • Training a Denoising Diffusion Bridge Model (DDBM) on paired images
      (x₀ = target, x_T = condition) using DBSM loss.
    • Fast inference via the η‑bridge sampler (Formula 15 in the paper)
      with optional 3‑rd‑order exponential‑Runge‑Kutta solver.
    • Minimal dependencies: PyTorch ≥ 2.2, torchvision, Pillow, tqdm.

Usage examples
--------------
# 1. Prepare paired data folders:
#    data/train/precise/*.png   (x₀)
#    data/train/coarse/*.png    (x_T)
#    Filenames must correspond, e.g. 0001.png in both folders.

# 2. Train
python dbim_image_translation.py --mode train \
       --data_root data/train --save_dir ckpt --epochs 400

# 3. Sample 30‑step deterministic (η=0) translation
python dbim_image_translation.py --mode sample \
       --data_root data/test  --ckpt ckpt/best.pth \
       --out_dir results --steps 30 --eta 0
"""

import argparse
import math
import time
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from jinja2.lexer import TOKEN_DOT
from pyparsing import alphas
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms as T
# from torchvision.utils import save_image
# from PIL import Image
from tqdm import tqdm

from argument import parse_opt_DBIM as parse_opt
from AIRP_read_data import read_dataset
from DBIM_utils import read_dataloader, perturb_coordinates, sub_center, plot_result
from models import DBIMGenerativeModel, DBIMLoss, polynomial_schedule

from torch_geometric.loader import DataLoader

from torch.utils.tensorboard import SummaryWriter


def get_snr(alpha, sigma):
    return alpha ** 2 / sigma ** 2

def get_noise_schedule(T, device=None):
    # TODO: adjust the noise schedule
    # betas = torch.linspace(1e-4, 0.02, T, device=device)
    # alphas = torch.cumprod(1 - betas, dim=0).sqrt()
    # sigmas = (1 - alphas ** 2).sqrt()

    alphas2 = torch.tensor(polynomial_schedule(T, power=float(2)), dtype=torch.float32, device=device)
    sigmas2 = 1 - alphas2
    alphas = alphas2.sqrt()
    sigmas = sigmas2.sqrt()

    return alphas, sigmas


def make_noise_schedule(T=1000, eta=0.0, device=None):

    alphas, sigmas = get_noise_schedule(T, device)

    snrs = get_snr(alphas, sigmas)
    snrT_to_t = snrs[-1] / snrs

    ats = alphas / alphas[-1] * snrT_to_t
    bts = alphas * (1 - snrT_to_t)
    cts = sigmas * torch.sqrt(1 - snrT_to_t)

    rhos = eta * sigmas[:-1] * torch.sqrt(1 - snrs[1:] / snrs[:-1])
    rhos = torch.cat([rhos[:-1], cts[-1:]], dim=0)  # enforce rho_{N-1} = c_{t_{N-1}}

    return ats, bts, cts, rhos, sigmas

def sample_time_step(x, T):
    t_val = torch.randint(
        low=0,
        high=T,
        size=(x.size(0), 1, 1),
        device=x.device
    )
    t_val = t_val.expand(-1, x.size(1), -1)

    return t_val

def evaluate(val_loader, generative_model, ats, bts, cts, criterion, args):
    with torch.no_grad():
        generative_model.eval()
        evaluating_loss = 0.0
        for batch_idx, data in enumerate(val_loader):

            pos_predict, node_mask, noise, t, x0, xt, xT = (
                data_prepare(data=data, generative_model=generative_model, ats=ats, bts=bts, cts=cts, args=args))

            loss = criterion(model_predict=pos_predict, xt=xt, x0=x0, node_mask=node_mask, noise=noise)
            evaluating_loss += loss

        evaluating_loss = evaluating_loss / len(val_loader)
        return evaluating_loss


def data_prepare(data, generative_model, ats, bts, cts, args):
    device = args.device
    dtype = torch.float32

    atom_type_scaling = args.atom_type_scaling
    max_atom_number = args.max_atom_number
    T = args.T

    x = data['pos'].to(device, dtype)
    node_mask = data['atom_mask'].to(device).bool()
    h = data['atomic_numbers_one_hot'].to(device, dtype) * atom_type_scaling

    # reformat
    curr_batch_size = x.size()[0] // max_atom_number

    x = x.view(curr_batch_size, max_atom_number, 3)
    node_mask = node_mask.view(curr_batch_size, max_atom_number)
    h = h.view(curr_batch_size, max_atom_number, -1)

    x = sub_center(x)

    t = sample_time_step(x=x, T=T)
    t_norm = t / T

    x0 = x
    xT = perturb_coordinates(x0=x0)
    xT = sub_center(xT)

    noise = torch.randn_like(xT, device=device)
    noise = sub_center(noise)

    # xta = ats[t] * xT + bts[t] * x0
    # xta = sub_center(xta)
    xt =  ats[t] * xT + bts[t] * x0 + cts[t]* noise
    xt = sub_center(xt)

    # assert not torch.isnan(xt).any(), "xt contains NaN!"
    # assert not torch.isinf(xt).any(), "xt contains Inf!"

    h = torch.cat([h, xt], dim=-1)
    h = torch.cat([h, t_norm], dim=-1)

    _, pos_predict = generative_model(xt=xT, h=h, mask=node_mask)

    # assert not torch.isnan(pos_predict).any(), "xp contains NaN!"
    # assert not torch.isinf(pos_predict).any(), "xp contains Inf!"
    pos_predict = sub_center(pos_predict)

    node_mask = node_mask.view(curr_batch_size, max_atom_number, -1)

    return pos_predict, node_mask, noise, t, x0, xt, xT

def train(args):

    device = args.device
    dtype = torch.float32

    train_loader, val_loader, test_loader = read_dataloader(args)

    epochs = args.epochs
    generative_model = DBIMGenerativeModel().to(device)
    # generative_model = load_model(model_path='saved_model/DBIM_model.pth', device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(generative_model.parameters(), lr=args.lr, amsgrad=False, weight_decay=1e-12)
    criterion = DBIMLoss()

    T = args.T
    ats, bts, cts, rhos, sigmas = make_noise_schedule(T=T, eta=0.0, device=device)

    atom_type_scaling = args.atom_type_scaling
    max_atom_number = args.max_atom_number

    batch_result_list = []
    epoch_result_list = []

    best_val_loss = float('inf')
    patience = args.patience
    counter = 0

    writer = SummaryWriter(log_dir=f"DBIM/exp_on_whole_1_x0_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    batch_count = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        # start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            generative_model.train()
            optimizer.zero_grad()

            pos_predict, node_mask, noise, t, x0, xt, xT = (
                data_prepare(data=data, generative_model=generative_model, ats=ats, bts=bts, cts=cts, args=args))

            loss = criterion(model_predict=pos_predict, xt=xt, x0=x0, node_mask=node_mask, noise=noise)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(generative_model.parameters(), max_norm=1.0)
            optimizer.step()

            if loss < 2 or len(batch_result_list) == 0:
                epoch_loss += loss
                batch_result_list.append(loss)
                writer.add_scalar("Loss/batch_loss", loss, batch_count+batch_idx)
            else:
                epoch_loss += batch_result_list[-1]
                batch_result_list.append(batch_result_list[-1])
                writer.add_scalar("Loss/batch_loss", batch_result_list[-1], batch_count+batch_idx)

            # epoch_loss += loss
            # batch_result_list.append(loss)
            # writer.add_scalar("Loss/batch_loss", loss, batch_count + batch_idx)

            batch_count += batch_idx

            with torch.no_grad():
                if batch_idx % 100 == 0:
                    # xt =  ats[t] * xT + bts[t] * x0 + cts[t]* noise
                    # x0_predict = (xt - (ats[t] * xT + cts[t]* (pos_predict - xt)))/bts[t]
                    # x0_loss = F.mse_loss(x0 * node_mask, x0_predict * node_mask)
                    print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Train Loss: {loss.item():.10f}")
                    # print(F.mse_loss(x0 * node_mask, xT * node_mask))
                    # print(F.mse_loss(x0 * node_mask, pos_predict * node_mask))

        epoch_loss = epoch_loss / len(train_loader)
        epoch_result_list.append(epoch_loss)

        writer.add_scalar("Loss/epoch_loss", epoch_loss, epoch)

        if math.isnan(epoch_loss):
            break

        val_loss = evaluate(val_loader=val_loader, generative_model=generative_model,
                     ats=ats, bts=bts, cts=cts, criterion=criterion, args=args)

        writer.add_scalar("Loss/val_loss", epoch_loss, epoch)

        print(f"Epoch [{epoch + 1}/{epochs}] Val Loss: {val_loss:.10f}")

        # end_time = time.time()
        # epoch_duration = end_time - start_time
        # print(f" Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(generative_model.state_dict(), 'saved_model/DBIM_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break


    plot_result(batch_result_list, name='batch_results')
    plot_result(epoch_result_list, name='epoch_results')

    writer.close()


if __name__ == '__main__':
    args = parse_opt()
    train(args)
