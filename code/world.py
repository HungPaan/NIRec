from parse import parse_args
import ast
import numpy as np

args = parse_args()
config = {}


config['data_path'] = args.data_path
config['data_name'] = args.data_name
config['groundtruth_para_path'] = args.groundtruth_para_path
config['best_para_path'] = args.best_para_path
config['metric_path'] = args.metric_path
config['completion_type'] = args.completion_type
config['completion_dim'] = args.completion_dim

config['sample_type'] = args.sample_type
config['is_rm'] = args.is_rm
config['filter_model_type'] = args.filter_model_type
config['prefer_thres'] = args.prefer_thres
config['num_neigh_thres'] = args.num_neigh_thres
config['group_size'] = args.group_size
config['num_group'] = args.num_group
config['num_guided_item'] = args.num_guided_item
config['baseline_reward_type'] = args.baseline_reward_type
config['base_penalty_type'] = args.base_penalty_type
config['num_base_rec_per_user'] = args.num_base_rec_per_user

config['base_model_name'] = args.base_model_name
config['click_model_name'] = args.click_model_name
config['rm_click_model_name'] = args.rm_click_model_name

config['guided_users'] = np.array(ast.literal_eval(args.guided_users))
config['guided_item'] = args.guided_item

config['emb_dim'] = args.emb_dim
config['num_layers'] = args.num_layers

config['alpha1'] = args.alpha1
config['alpha2'] = args.alpha2
config['alpha3'] = args.alpha3
config['indi_scale'] = args.indi_scale
config['neigh_scale'] = args.neigh_scale
config['att_scale'] = args.att_scale
config['beta'] = args.beta
config['power'] = args.power

config['penalty_weight'] = args.penalty_weight

config['device'] = args.device
config['seed'] = args.seed
config['core'] = args.core
config['val_ratio'] = args.val_ratio
