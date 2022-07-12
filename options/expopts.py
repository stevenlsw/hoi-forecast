def add_exp_opts(parser):
    parser.add_argument("--resume", type=str, nargs="+", metavar="PATH",
                        help="path to latest checkpoint (default: none)")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true",
                        help="evaluate model on validation set")
    parser.add_argument("--test_freq", type=int, default=10,
                        help="testing frequency on evaluation dataset (set specific in traineval.py)")
    parser.add_argument("--snapshot", default=10, type=int, metavar="N",
                        help="How often to take a snapshot of the model (0 = never)")
    parser.add_argument("--use_cuda", default=1, type=int, help="use GPU (default: True)")
    parser.add_argument('--ek_version', default="ek55", choices=["ek55", "ek100"], help="epic dataset version")
    parser.add_argument("--traj_only", action="store_true", help="evaluate traj on validation dataset")