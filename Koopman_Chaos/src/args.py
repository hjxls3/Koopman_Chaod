import argparse
import errno
import json
import os


def mkdirs(*directories):
    for directory in list(directories):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Federated Koopman')
        # System settings
        self.add_argument('--exp_name', type=str, default='AI4Chaotic', help='the name of the experiment')
        self.add_argument('--res_dir', type=str, default='./results',
                          help='directory to save experiment results')
        self.add_argument('--num_trajectories', type=int, default=10,
                          help='number of trajectories as training data')

        # Dynamical system
        self.add_argument('--dyn_sys_name', type=str, default='Lorenz63',
                          help='name of the dynamical system')
        self.add_argument('--dim_dyn_sys', type=int, default=3, help='dimensions of dynamical system')
        self.add_argument('--t_span_end', type=float, default=10,
                          help='time interval for the ode solver')


        # Lorenz 63
        self.add_argument('--sigma', type=float, default=10, help='sigma in Lorenz63')
        self.add_argument('--rho', type=float, default=28, help='rho in Lorenz63')
        self.add_argument('--beta', type=float, default=8/3, help='beta in Lorenz63')
        self.add_argument('--init_high', type=float, default=10,
                          help="upper bound of initial condition in Lorenz63 system")
        self.add_argument('--init_low', type=float, default=-10,
                          help="lower bound of initial condition in Lorenz63 system")

        # Koopman Network
        self.add_argument('--koopman_obs_dim', type=int, default=12, help='koopman observable dimension')
        self.add_argument('--koopman_hidden_dim', type=int, default=24,
                          help='koopman autoencoder hidden layer dimension')

        # learning
        self.add_argument('--train_test_ratio', type=float, default=0.9,
                          help='the ratio of training data in all data, the rest is my_test data')
        self.add_argument('--input_size', type=int, default=10, help='number of time-steps as input')
        self.add_argument('--batch_size', type=int, default=64, help='batch size for training')
        self.add_argument('--epoch', type=int, default=100, help='training epochs for each local learn')
        self.add_argument('--koopman_lr', type=float, default=0.005, help='learning rate for koopman')
        self.add_argument('--koopman_lr_scheduler_gamma', type=float, default=0.995,
                          help='gamma in lr scheduler for koopman')
        self.add_argument('--koopman_adam_weight_decay', type=float, default=1e-7,
                          help='weight decay in Adam optimizer for koopman')

        # logging
        self.add_argument('--notes', type=str, required=True,
                          help='notes to distinguish between different tests')

    def parse(self, dirs=True):
        args = self.parse_args()

        args.run_dir = args.res_dir + '/{}_{}'.format(args.exp_name, args.notes)

        if dirs:
            mkdirs(args.run_dir)

        if dirs:
            with open(args.run_dir + "/args.json", 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        return args
