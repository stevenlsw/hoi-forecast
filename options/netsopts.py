def add_nets_opts(parser):
    parser.add_argument('--src_in_features', type=int, default=1024, help='Network encoder input size')
    parser.add_argument('--trg_in_features', type=int, default=2, help='Network decoder input size')
    parser.add_argument('--num_patches', type=int, default=5, help='Number of classes')
    parser.add_argument('--num_classes', type=int, default=2513, help='Number of classes')

    parser.add_argument('--embed_dim', type=int, default=512, help='embedded dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='num of heads in transformer')
    parser.add_argument('--enc_depth', type=int, default=6, help='transformer encoder depth')
    parser.add_argument('--dec_depth', type=int, default=4, help='transformer decoder depth')

    parser.add_argument('--coord_dim', type=int, default=64, help='coordinates feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='stochastic modules hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=256, help='stochastic modules latent dimension')

    parser.add_argument("--encoder_time_embed_type", default="sin",
                        choices=["sin", "param"], help="transformer encoder time position embedding")
    parser.add_argument("--decoder_time_embed_type", default="sin",
                        choices=["sin", "param"], help="transformer decoder time position embedding")

    parser.add_argument("--num_samples", default=20, type=int, help="get number of samples during inference, "
                                                                    "stochastic model multiple runs")
    parser.add_argument("--num_points", default=5, type=int,
                        help="number of remaining contact points after farthest point "
                             "sampling for evaluation affordance")
    parser.add_argument("--gaussian_sigma", default=3., type=float,
                        help="predicted contact points gaussian kernel sigma")
    parser.add_argument("--gaussian_k_ratio", default=3., type=float,
                        help="predicted contact points gaussian kernel size")

    # Options for loss supervision
    parser.add_argument("--lambda_obj", default=1e-1, type=float, help="Weight to supervise object affordance")
    parser.add_argument("--lambda_traj", default=1., type=float, help="Weight to supervise hand traj")
    parser.add_argument("--lambda_obj_kl", default=1e-3, type=float, help="Weight to supervise object affordance KLD")
    parser.add_argument("--lambda_traj_kl", default=1e-3, type=float, help="Weight to supervise hand traj KLD")


def add_train_opts(parser):
    parser.add_argument("--manual_seed", default=0, type=int, help="manual seed")
    parser.add_argument("-j", "--workers", default=16, type=int, help="number of workers")
    parser.add_argument("--epochs", default=35, type=int, help="number epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")

    parser.add_argument("--optimizer", default="adam", choices=["rms", "adam", "sgd", "adamw"])
    parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float, metavar="LR", help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float)

    parser.add_argument("--scheduler", default="cosine", choices=['cosine', 'step', 'multistep'],
                        help="learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs to run")
    parser.add_argument("--lr_decay_step", nargs="+", default=10, type=int,
                        help="Epochs after which to decay learning rate")
    parser.add_argument(
        "--lr_decay_gamma", default=0.5, type=float, help="Factor by which to decay the learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float)
