import argparse


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def dict2namespace(config):
	namespace = argparse.Namespace()
	for key, value in config.items():
		if isinstance(value, dict):
			new_value = dict2namespace(value)
		else:
			new_value = value
		setattr(namespace, key, new_value)
	return namespace