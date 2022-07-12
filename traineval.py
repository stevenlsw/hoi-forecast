import argparse
import os
import random

import numpy as np
import torch
import torch.nn.parallel
import torch.optim

from netscripts.get_datasets import get_dataset
from netscripts.get_network import get_network
from netscripts.get_optimizer import get_optimizer
from netscripts import modelio
from netscripts.epoch_feat import epoch_pass
from options import netsopts, expopts
from datasets.datasetopts import DatasetArgs


def main(args):

    # Initialize randoms seeds
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    datasetargs = DatasetArgs(ek_version=args.ek_version)
    num_frames_input = int(datasetargs.fps * datasetargs.t_buffer)
    num_frames_output = int(datasetargs.fps * datasetargs.t_ant)
    model = get_network(args, num_frames_input=num_frames_input,
                        num_frames_output=num_frames_output)

    if args.use_cuda and torch.cuda.is_available():
        print("Using {} GPUs !".format(torch.cuda.device_count()))
        model.cuda()

    start_epoch = 0
    if args.resume is not None:
        device = torch.device('cuda') if torch.cuda.is_available() and args.use_cuda else torch.device('cpu')
        start_epoch = modelio.load_checkpoint(model, resume_path=args.resume[0], strict=False, device=device)
        print("Loaded checkpoint from epoch {}, starting from there".format(start_epoch))

    _, dls = get_dataset(args, base_path="./")

    if args.evaluate:
        args.epochs = start_epoch + 1
        traj_val_loader = None
    else:
        train_loader = dls['train']
        traj_val_loader = dls['validation']
        print("training dataset size: {}".format(len(train_loader.dataset)))
        optimizer, scheduler = get_optimizer(args, model=model, train_loader=train_loader)

    if not args.traj_only:
        val_loader = dls['eval']
    else:
        traj_val_loader = val_loader = dls['validation']
    print("evaluation dataset size: {}".format(len(val_loader.dataset)))

    for epoch in range(start_epoch, args.epochs):
        if not args.evaluate:
            print("Using lr {}".format(optimizer.param_groups[0]["lr"]))
            epoch_pass(
                loader=train_loader,
                model=model,
                phase='train',
                optimizer=optimizer,
                epoch=epoch,
                train=True,
                use_cuda=args.use_cuda,
                scheduler=scheduler)

        if args.evaluate or (epoch + 1) % args.test_freq == 0:
            with torch.no_grad():
                if not args.traj_only:
                    epoch_pass(
                        loader=val_loader,
                        model=model,
                        epoch=epoch,
                        phase='affordance',
                        optimizer=None,
                        train=False,
                        use_cuda=args.use_cuda,
                        num_samples=args.num_samples,
                        num_points=args.num_points)
                else:
                    with torch.no_grad():
                        epoch_pass(
                            loader=traj_val_loader,
                            model=model,
                            epoch=epoch,
                            phase='traj',
                            optimizer=None,
                            train=False,
                            use_cuda=args.use_cuda,
                            num_samples=args.num_samples)

        if not args.evaluate:
            if (epoch + 1 - args.warmup_epochs) % args.snapshot == 0:
                print(f"save epoch {epoch+1} checkpoint to {os.path.join(args.host_folder,args.exp_id)}")
                modelio.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "network": args.network,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint=os.path.join(args.host_folder, args.exp_id),
                filename = f"checkpoint_{epoch+1}.pth.tar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOI Forecasting")
    netsopts.add_nets_opts(parser)
    netsopts.add_train_opts(parser)
    expopts.add_exp_opts(parser)
    args = parser.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        args.batch_size = args.batch_size * num_gpus
        args.lr = args.lr * num_gpus

    if args.traj_only: assert args.evaluate, "evaluate trajectory on validation set must set --evaluate"
    main(args)
    print("All done !")