from parse import parse_args
args = parse_args()
config = {}


config['data_path'] = args.data_path
config['data_name'] = args.data_name
config['loss_type'] = args.loss_type

config['device'] = args.device
config['max_epochs'] = args.max_epochs

config['seed'] = args.seed
config['tensorborad_path'] = args.tensorborad_path
config['is_tensorboard'] = args.is_tensorboard

config['emb_model_name'] = args.emb_model_name
config['ips_type'] = args.ips_type
config['power'] = args.power

config['emb_dim'] = args.emb_dim
config['emb_lr'] = args.emb_lr
config['emb_decay'] = args.emb_decay

config['val_ratio'] = args.val_ratio
config['core'] = args.core

