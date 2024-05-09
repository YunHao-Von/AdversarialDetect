import argparse
import logging
import utils
import yaml
import os


from utils import str2bool


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, default='imagenet.yml',  help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1235, help='Random seed')
    parser.add_argument('--exp', type=str, default='./exp_results', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='imagenet', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=1000, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='sde', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='score_sde', help='[guided_diffusion, score_sde]')
    parser.add_argument('--epsilon', default=0.01568, type=float,help='perturbation')#0.01568, type=float,help='perturbation') 4/255
    parser.add_argument('--num-steps', default=5, type=int,help='perturb number of steps')
    
    args = parser.parse_args()
    args.step_size_adv = args.epsilon / args.num_steps
    
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)
    
    level = getattr(logging, args.verbose().upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))
    
    handler1 = logging.StreamHandler()
    formatter = logging.Formatter()
    

def robustness_eval():
    pass
if __name__ == '__main__':
    args, config = parse_args_and_config()