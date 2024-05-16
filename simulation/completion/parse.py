import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="simulation")
    parser.add_argument('--data_path', type=str, default='../../datasets/', help="data path")
    parser.add_argument('--data_name', type=str, default='ciao', help="data name")
    parser.add_argument('--loss_type', type=str, default='mse', help="mse, mat_mse")

    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--max_epochs', type=int, default=2000, help='the number of iterations')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--tensorborad_path', type=str, default='../tensorboards/',
                        help='tensorboard path')
    parser.add_argument('--is_tensorboard', type=int, default=1, help='is tensorboard')

    parser.add_argument('--emb_model_name', type=str, default='mf', help='mf')
    parser.add_argument('--ips_type', type=str, default='logi', help='logi, pop')
    parser.add_argument('--power', type=float, default=1.0, help='0.0, 0.25, 0.5, 0.75, 1.0')

    parser.add_argument('--emb_dim', type=int, default=32, help='dimension of all embeddings')
    parser.add_argument('--emb_lr', type=float, default=0.005, help='learning rate for learning emb_model')
    parser.add_argument('--emb_decay', type=float, default=0.0, help='weight decay for learning emb_model')

    parser.add_argument('--val_ratio', type=float, default=0.1, help='the proportion of validation set')
    parser.add_argument('--core', type=int, default=10, help='k-core')

    return parser.parse_args()

