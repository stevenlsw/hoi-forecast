import torch


class Warmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            init_lr_ratio: float = 0.0,
            num_epochs: int = 5,
            last_epoch: int = -1,
            iters_per_epoch: int = None,
):
        self.base_scheduler = scheduler
        self.warmup_iters = max(num_epochs * iters_per_epoch, 1)
        if self.warmup_iters > 1:
            self.init_lr_ratio = init_lr_ratio
        else:
            self.init_lr_ratio = 1.0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        assert self.last_epoch < self.warmup_iters
        return [
            el * (self.init_lr_ratio + (1 - self.init_lr_ratio) *
                  (float(self.last_epoch) / self.warmup_iters))
            for el in self.base_lrs
        ]

    def step(self, *args, **kwargs):
        if self.last_epoch < (self.warmup_iters - 1):
            super().step(*args, **kwargs)
        else:
            self.base_scheduler.step(*args, **kwargs)


def get_optimizer(args, model, train_loader):
    assert train_loader is not None, "train_loader is None, " \
                                     "warmup or cosine learning rate need number of iterations in dataloader"
    iters_per_epoch = len(train_loader)
    vae_params = [p for p_name, p in model.named_parameters()
                  if ('vae' in p_name or 'head' in p_name) and p.requires_grad]
    other_params = [p for p_name, p in model.named_parameters()
                    if ('vae' not in p_name and 'head' not in p_name) and p.requires_grad]

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam([{'params': vae_params, 'weight_decay': 0.0}, {'params': other_params}],
                                     lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "rms":
        optimizer = torch.optim.RMSprop([{'params': vae_params, 'weight_decay': 0.0}, {'params': other_params}],
                                        lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD([{'params': vae_params, 'weight_decay': 0.0}, {'params': other_params}],
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW([{'params': vae_params, 'weight_decay': 0.0}, {'params': other_params}], lr=args.lr)
    else:
        raise ValueError("unsupported optimizer type")

    for group in optimizer.param_groups:
        group["lr"] = args.lr
        group["initial_lr"] = args.lr

    if args.scheduler == "step":
        assert isinstance(args.lr_decay_step, int), "learning rate scheduler need integar lr_decay_step"
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma)
    elif args.scheduler == "multistep":
        if isinstance(args.lr_decay_step, list):
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.epochs // 2, gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*iters_per_epoch,
                                                               last_epoch=-1, eta_min=0)
    else:
        raise ValueError("Unrecognized learning rate scheduler {}".format(args.scheduler))

    main_scheduler = Warmup(optimizer, scheduler, init_lr_ratio=0., num_epochs=args.warmup_epochs,
                            iters_per_epoch=iters_per_epoch)
    return optimizer, main_scheduler




