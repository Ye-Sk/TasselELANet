"""
@author: Jianxiong Ye
"""

import math
import time
import yaml
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from torch.optim import lr_scheduler

import val as validate
from models.decoder import Model
from models.utils.loss import ComputeLoss
from models.utils.plot import plot_results
from models.utils.dataset import create_dataloader
from models.utils.helper import verify_img_size, logger, colorstr, print_info, crement_path
from models.utils.traineval import ModelEMA, check_dataset, EarlyStopping, init_optimizer, set_seed, calculate_class_weights, finalize_model_training


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='data/weights/TasselELANet.pt', help='transfer learning weights path')
    parser.add_argument('--cfg', type=str, default='config/TasselELANet.yaml', help='model.yaml path')
    parser.add_argument('--imgsz', type=int, default=608, help='train image size (pixels)')
    parser.add_argument('--data', type=str, default='config/dataset/MTDC.yaml', help='config.yaml path')
    parser.add_argument('--epochs', type=int, default=300, help='total iteration count')
    parser.add_argument('--batch-size', type=int, default=4, help='image batch size')
    parser.add_argument('--workers', type=int, default=2, help='max dataloader workers')
    parser.add_argument('--cache', type=bool, default=True, help='cache images for faster training')
    parser.add_argument('--cos-lr', action='store_true', default=False, help='cosine LR scheduler')
    parser.add_argument('--patience', type=int, default=50, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--seed', type=int, default=2023, help='Global training seed')
    opt = parser.parse_args()
    print_info(vars(opt))
    return opt

def trainer(opt):
    hyp, save_dir, epochs, batch_size, weights, data, cfg, workers = \
    opt.data, Path(crement_path('runs/train/exp')), opt.epochs, opt.batch_size, opt.pretrained, opt.data, opt.cfg, opt.workers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Hyperparameters
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    cuda = device.type != 'cpu'

    # Save run settings
    with open(save_dir / 'config.txt', 'w') as f:
        f.write('Data:\n')
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in hyp.items()}, f, sort_keys=False)
        f.write('\nOpt:\n')
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    set_seed(opt.seed)  # set random seed

    data_dict = check_dataset(data)

    train_path, val_path, nc, names = data_dict['train'], data_dict['train'], int(data_dict['nc']), data_dict['names']

    try:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = {k: v for k, v in csd.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(csd, strict=False)  # load
        logger.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        del ckpt, csd
    except:
        model = Model(cfg, ch=3, nc=nc).to(device)  # create
        logger.warning(f'WARNING ⚠️ --Failed to initialize weights: {weights}')

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = verify_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Optimizer
    nbs, accumulate = 64, max(round(64 / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = init_optimizer(model, hyp['optimizer'], hyp['ilr'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = lambda x: (1 - math.cos(x * math.pi / epochs)) / 2 * 0.99 + 1  # cosine
    else:
        lf = lambda x: (1 - x / epochs) * 0.99 + 0.01  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model)

    best_fitness, start_epoch = 0.0, 0

    # Trainloader (augment)
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, hyp=hyp, augment=True,
                                              cache=opt.cache, workers=workers, prefix=colorstr('train: '), shuffle=True)

    # Valloader (no augment)
    val_loader = create_dataloader(val_path, imgsz, batch_size, gs, hyp=hyp, cache=opt.cache,
                                   workers=workers * 2, prefix=colorstr('val: '))[0]

    model.half().float()  # pre-reduce anchor precision

    # Model attributes
    model.nc, model.hyp, model.names = nc, hyp, names
    model.class_weights = calculate_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(3.0 * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=cuda)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Starting training for {epochs} epochs...')

    titles = ["Epoch", "GPU_mem", "box_loss", "cls_loss", "dfl_loss", "Instances", "Size", "P ", "R  ", "mAP50   ", "mAP50-95   "]
    title_line = ''.join(f'{title:>11s}' for title in titles)
    with open(results_file, 'a') as f:
        f.write(title_line + '\n')

    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        logger.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'cls_loss', 'dfl_loss', 'Instances', 'Size'))
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}')  # progress bar
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255

            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.8, hyp['momentum']])

            with torch.cuda.amp.autocast(enabled=cuda):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))

            scaler.scale(loss).backward()

            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            mloss = (mloss * i + loss_items) / (i + 1)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            info = ('%11s' * 2 + '%11.4g' * 5) % (
                f'{epoch +1 }/{epochs}', mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(info)

        scheduler.step()

        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
        results, maps = validate.run(data_dict,
                                        batch_size=batch_size,
                                        imgsz=imgsz,
                                        model=ema.ema,
                                        dataloader=val_loader,
                                        save_dir=save_dir,
                                        plots=False,
                                        compute_loss=compute_loss)

        with open(results_file, 'a') as f:
            f.write(info + '%10.4g' * 4 % results[:-3] + '\n')

        fi = (np.array(results).reshape(1, -1)[:, :4] * [0.0, 0.0, 0.1, 0.9]).sum(1)
        stop = stopper(epoch=epoch, fitness=fi)
        if fi > best_fitness:
            best_fitness = fi

        if final_epoch or best_fitness == fi:
            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'model': deepcopy(model).half(),
                'ema': deepcopy(ema.ema).half(),
                'updates': ema.updates,
                'optimizer': optimizer.state_dict(),
                'opt': vars(opt),}

            torch.save(ckpt, best)
            torch.save(ckpt, last)
            del ckpt

        if stop:
            break

    logger.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    for f in last, best:
        if f.exists():
            finalize_model_training(f)  # strip optimizers
    plot_results(save_dir)
    logger.info(f"Training results saved to {colorstr('bold', save_dir)}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # Ignore any warnings
    opt = parse_opt()
    trainer(opt)
