import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="group policy optimization var2")
    parser.add_argument('--data_name', type=str, default='ciao', help="name of data")
    parser.add_argument('--data_path', type=str, default='./datasets/', help="path of data")
    parser.add_argument('--groundtruth_para_path', type=str, default='./simulation/para/',
                        help="path of groundtruth model parameters")
    parser.add_argument('--best_para_path', type=str, default='./best_para_path/',
                        help="path of click model parameters")
    parser.add_argument('--metric_path', type=str, default='./metric/',
                        help="path of metrics")
    parser.add_argument('--completion_type', type=str, default='full_mf', help="completion type")
    parser.add_argument('--completion_dim', type=int, default=20, help="completion model dimension")


    parser.add_argument('--filter_model_type', type=str, default='groundtruth',
                        help='type of constructing the guided population')
    parser.add_argument('--sample_type', type=str, default='ph', help='ph')
    parser.add_argument('--is_rm', type=int, default=0, help='')
    parser.add_argument('--prefer_thres', type=float, default=1.0, help='')
    parser.add_argument('--num_neigh_thres', type=int, default=0, help='')
    parser.add_argument('--group_size', type=int, default=1, help='#unit in a group')
    parser.add_argument('--num_group', type=int, default=1, help='#group')
    parser.add_argument('--num_guided_item', type=int, default=1000, help='#guided_item')
    parser.add_argument('--baseline_reward_type', type=str, default='spill',
                        help='type of reward for baseline (sub_mean, spill)')
    parser.add_argument('--base_penalty_type', type=str, default='auto',
                        help='type of penalty (auto, fix)')
    parser.add_argument('--num_base_rec_per_user', type=int, default=20,
                        help='the number of items recommended to each user')

    parser.add_argument('--base_model_name', type=str, default='lightgcn',
                        help='name of base model (lightgcn)')
    parser.add_argument('--click_model_name', type=str, default='groundtruth',
                        help='name of click model (groundtruth, gatv2_spill_var2, mf, diffnet)')
    parser.add_argument('--rm_click_model_name', type=str, default='groundtruth',
                        help='name of click model (groundtruth, gatv2_spill_var2, mf, diffnet)')

    parser.add_argument('--guided_users', type=str, default="[0]", help='guided users')
    parser.add_argument('--guided_item', type=int, default=99, help='guided item')


    parser.add_argument('--emb_dim', type=int, default=20, help='dim')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for diffnet/lightgcn')

    parser.add_argument('--alpha1', type=float, default=0.1, help='alpha1')
    parser.add_argument('--alpha2', type=float, default=0.2, help='alpha2')
    parser.add_argument('--alpha3', type=float, default=0.7, help='alpha3')
    parser.add_argument('--indi_scale', type=float, default=0.25, help='scale factor for individual preference')
    parser.add_argument('--neigh_scale', type=float, default=0.25, help='scale factor for neighbor preference')
    parser.add_argument('--att_scale', type=float, default=1.0, help='scale factor for attention')
    parser.add_argument('--beta', type=float, default=10.0, help='scale factor for neighbor effect')
    parser.add_argument('--power', type=float, default=0.5, help='normalization factor')

    parser.add_argument('--penalty_weight', type=float, default=0.0, help='weight')

    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--device', type=str, default='cuda', help='device (cpu, gpu)')
    parser.add_argument('--core', type=int, default=10, help='k-core')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='the proportion of validation set')


    return parser.parse_args()

