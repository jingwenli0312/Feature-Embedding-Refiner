import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        # description="the algorithm name"
    )

    ### overall settings
    parser.add_argument('--problem', default='cvrp', choices=['pdp', 'pdvrp', 'tsp', 'cvrp'],
                        help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=100,
                        help="The size of the problem graph")
    parser.add_argument('--eval_only', action='store_true',
                        help='used only if to evaluate a model')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='used only if to evaluate a model')
    parser.add_argument('--init_val_met', choices=['random', 'nearest'], default='random',
                        help='method to generate initial solutions while validation')
    parser.add_argument('--pretrained', default='./pretrained/cvrp_100/epoch-99.pt',
                        help='pretrained model from AM')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_tb', action='store_true', help='Disable Tensorboard')
    parser.add_argument('--no_figures', action='store_true', help='Disable figure logging')
    parser.add_argument('--no_saving', action='store_true', help='Disable saving checkpoints')
    parser.add_argument('--use_assert', action='store_true', help='Enable Assertion')
    parser.add_argument('--seed', type=int, default=4869, help='Random seed to use')
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    # new features
    parser.add_argument('--RL_agent', default='a2c', choices=['a2c', 'ppo'])
    parser.add_argument('--K', type=int, default=8, help='pool size')
    parser.add_argument('--step_method', default='insert', choices=['2_opt', 'swap', 'insert'])
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--use_real_mask', action='store_true')
    parser.add_argument('--critic_head_num', type=int, default=8)
    parser.add_argument('--encoder_head_num', type=int, default=8)
    parser.add_argument('--decoder_head_num', type=int, default=1)  # testing
    parser.add_argument('--P', type=int, default=20)  # perturb
    parser.add_argument('--multi_solu', type=int, default=4)

    # a2c params
    parser.add_argument('--K_epochs', type=int, default=3)
    parser.add_argument('--eps_clip', type=float, default=0.1)
    parser.add_argument('--reward_clip', type=int, default=1)  # problem_size
    parser.add_argument('--eps_range', type=float, default=0.005)
    parser.add_argument('--T_train', type=int, default=100)  # 250
    parser.add_argument('--n_step', type=int, default=4)

    # resume and load models
    parser.add_argument('--load_path', default=None,
                        help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', default=None,
                        help='Resume from previous checkpoint file')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')

    ### training AND validation
    parser.add_argument('--batch_size', type=int, default=512,  # 512 1024
                        help='Number of instances per batch during training')
    parser.add_argument('--epoch_end', type=int, default=100,
                        help='End at epoch #')
    parser.add_argument('--epoch_size', type=int, default=10240,  # 10 * batch_size
                        help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=1000,  # 1000
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=1000,  # 1000
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--val_dataset', type=str, default=None,
                        help='Dataset file to use for validation')

    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=5e-5, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.988, help='Learning rate decay per epoch')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')

    ### network
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_edge_dim', type=int, default=32, help='Dimension of input embedding')
    parser.add_argument('--feed_forward_dim', type=int, default=256, help='Dimension of feed forward layer')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--gamma', type=float, default=0.999, help='decrease future reward')
    parser.add_argument('--T_max', type=int, default=200, help='number of steps to swap')  # 1500

    ### logs to tensorboard and screen
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--log_step', type=int, default=10,  # 50
                        help='Log info every log_step steps')
    ### outputs
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--run_name', default='run_name', help='Name to identify the run')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    ) if not opts.no_saving else None
    return opts
