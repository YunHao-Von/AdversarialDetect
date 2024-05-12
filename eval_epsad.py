import argparse
import logging
import torch
import torch.nn as nn
import utils
import yaml
import os
import random
import numpy as np
import torch.nn.functional as F
import time


from utils import str2bool, get_image_classifier, load_detection_data

from runners.diffpure_ddpm import Diffusion
from runners.diffpure_guided import GuidedDiffusion
from runners.diffpure_sde import RevGuidedDiffusion
from runners.diffpure_ode import OdeGuidedDiffusion
from runners.diffpure_ldsde import LDGuidedDiffusion
from score_sde import sde_lib
from score_sde.models import utils as mutils
from ensattack import ens_attack


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, default='imagenet.yml',  help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1235, help='Random seed')
    parser.add_argument('--exp', type=str, default='./exp_results', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='imagenet', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='sde', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='score_sde', help='[guided_diffusion, score_sde]')
    parser.add_argument('--detection_datapath', type=str, default='./score_diffusion_t_imagenet_1w')#./score_diffusion_t_cifar
    parser.add_argument('--attack_type', type=str, default='square')
    parser.add_argument('--attack_version', type=str, default='standard')
    parser.add_argument('--classifier_name', type=str, default='imagenet-resnet50', help='which classifier to use')
    
    
    # diffusion models
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=1000, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--t_size', type=int,default=10)
    parser.add_argument('--datapath', type=str, default='/media/yunhao/DATA/AdversarialDetect/Data/ILSVRC2012_Data')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')
    
    # adv
    parser.add_argument('--domain', type=str, default='imagenet', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--adv_batch_size', type=int, default=16)

    # Detection
    # parser.add_argument('--generate_lw_flag', action='store_true')
    # parser.add_argument('--clean_score_flag', action='store_true')
    # parser.add_argument('--single_vector_norm_flag', action='store_true')
    parser.add_argument('--detection_ensattack_norm_flag', action='store_true')
    # parser.add_argument('--perb_image', action='store_true')
    parser.add_argument('--diffuse_t', type=int,default=50)	
    	
    
    # additional attack settings
    parser.add_argument('--num_sub', type=int, default=10000, help='imagenet subset')
    parser.add_argument('--attack_methods', type=str, nargs='+',default=['FGSM','FGSM_L2'])
    parser.add_argument('--num-steps', default=5, type=int,help='perturb number of steps')
    parser.add_argument('--epsilon', default=0.00392 , type=float,help='perturbation')#0.01568, type=float,help='perturbation')
    
    # Run1
    # parser.add_argument('--single_vector_norm_flag', type=str2bool, default=True)
    # parser.add_argument('--perb_image', type=str2bool, default=True)
    # parser.add_argument('--generate_lw_flag', type=str2bool, default=True)
    # parser.add_argument('--clean_score_flag', type=str2bool, default=True)
    # parser.add_argument('--gpu_ids', type=str, default='0,1')

    # Run2
    parser.add_argument('--single_vector_norm_flag', type=str2bool, default=True)
    parser.add_argument('--perb_image', type=str2bool, default=True)
    parser.add_argument('--generate_lw_flag', type=str2bool, default=True)
    parser.add_argument('--gpu_ids', type=str, default='0,1')
    parser.add_argument('--clean_score_flag', action='store_true')
    
    
    
    
    args = parser.parse_args()
    args.step_size_adv = args.epsilon / args.num_steps
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)
    
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))
    
    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)
    
    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)
    os.makedirs(args.detection_datapath, exist_ok=True)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info('Using device: {}'.format(device))
    new_config.device = device
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    
    return args, new_config
    
    

class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        
        self.classifier = get_image_classifier(args.classifier_name).to(config.device)
        
        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'sde':
            self.runner = RevGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ode':
            self.runner = OdeGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ldsde':
            self.runner = LDGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')
        
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None
    
    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)
    
    def set_tag(self, tag=None):
        self.tag = tag
    
    def forward(self,x):
        counter = self.counter.item()
        if counter %5 == 0:
            print(f"diffusion time: {counter}")
        
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
        
        start_time = time.time()
        if not args.detection_flag:
            x_re = self.runner.image_editing_sample((x-0.5)*2, bs_id=counter, tag=self.tag)
        else:
            x_re, ts_cat = self.runner.image_editing_sample((x-0.5)*2, bs_id=counter, tag=self.tag, t_size=self.args.t_size)
        minutes, seconds = divmod(time.time() - start_time, 60)
        
        if not args.detection_flag:
            if 'imagenet' in self.args.domain:
                x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)
            
            if counter % 5 == 0:
                print(f'x shape (before diffusion models): {x.shape}')
                print(f'x shape (before classifier): {x_re.shape}')
                print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))
            
            out = self.classifier((x_re + 1) * 0.5)
            
        self.counter += 1
        return out if not args.detection_flag else x_re, ts_cat
        
        

def robustness_eval(args, config):
    middle_name = '_'.join([args.diffusion_type, args.attack_version]) if args.attack_version in ['stadv', 'standard', 'rand'] \
        else '_'.join([args.diffusion_type, args.attack_version, args.attack_type])
    
    log_dir = os.path.join(args.image_folder, args.classifier_name, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = utils.Logger(file_name=f"{log_dir}/log.txt", file_mode='w+', should_flush=True)
    
    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')
    print('starting the model and loader...')
    
    model = SDE_Adv_Model(args, config)
    if ngpus > 1:
        model = torch.nn.DataParallel(model)
    # model = model.eval().to(config.device)
    model = model.eval()
    
    loader = load_detection_data(args, adv_batch_size, args.generate_lw_flag)
    
    detection_test_ensattack(args, config, model, loader)
    
    logger.close()    
    
    
    

def detection_test_ensattack(args, config, model ,loader):
    ngpus = torch.cuda.device_count()
    model_ = model
    if ngpus > 1:
        model_ = model.module
    
    print(f'attack_list: {args.attack_methods}') 
    print(f'num_steps: {args.num_steps}, epsilon: {args.epsilon}')
    classifier = get_image_classifier(args.classifier_name).to(config.device)
    
    if config.data.dataset == 'ImageNet':
        args.num_classes = 1000
        mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).view(1,3,1,1).cuda().float()
        std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).view(1,3,1,1).cuda().float()
    
    rev_vpsde = model_.runner.rev_vpsde
    sde = sde_lib.VPSDE(beta_min=rev_vpsde.beta_0, beta_max=rev_vpsde.beta_1, N=rev_vpsde.N)
    score_fn = mutils.get_score_fn(sde, rev_vpsde.model, train=False, continuous=True)
    attack_methods = args.attack_methods
    if args.clean_score_flag:
        attack_methods = ['no_attack']
    
    print('reset the mean 0 and std 1')
    mean = mean - mean
    std = std / std
    if args.detection_ensattack_norm_flag:
        print('you are calulating the norm of score!')
    if args.single_vector_norm_flag:
        print('you are calulating the single_vector_norm of score!')
    
    score_adv_list = []
    score_adv_lists = []
    assert not (args.detection_ensattack_norm_flag==True and args.single_vector_norm_flag==True)
    isnorm = '_norm' if args.detection_ensattack_norm_flag else ''
    isperb_image = 'perb_image' if args.perb_image else ''
    data_size = '' if args.num_sub==500 else str(args.num_sub)
    
    for attack_method in attack_methods:
        top1_counter = 0
        top5_counter = 0
        num_samples = 0
        start_time = time.time()
        j = -1
        for i, (x, y) in enumerate(loader):
            x = x.to(config.device)
            y = y.to(config.device)
            
            if not args.clean_score_flag:
                if attack_method == 'MM_Attack' or attack_method == 'AA_Attack':
                    with torch.no_grad():
                        output = classifier(x)
                    correct_batch = y.eq(output.max(dim=1)[1])
                    x = x[correct_batch]
                    y = y[correct_batch]
                x_adv, top1, top5 = ens_attack(x, y, classifier, mean, std, args, attack_method)
            else:
                x_adv = x
                top1, top5 = 1, 1
            
            top1_counter += top1 * x.shape[0]
            top5_counter += top5 * x.shape[0]
            num_samples += x.shape[0]
            score_sum = torch.zeros_like(x_adv, device=x_adv.device)
            with torch.no_grad():
                for value in range(1,args.diffuse_t+1):
                    t_valuve = value/1000
                    curr_t_temp = torch.tensor(t_valuve,device=x.device)
                    if args.perb_image:
                        z = torch.randn_like(x_adv, device=x_adv.device)
                        mean_x_adv, std_x_adv = sde.marginal_prob(2*x_adv-1, curr_t_temp.expand(x_adv.shape[0]))
                        perturbed_data = mean_x_adv + std_x_adv[:, None, None, None] * z
                        score = score_fn(perturbed_data, curr_t_temp.expand(x_adv.shape[0]))
                    
                    else:
                        score = score_fn(2*x_adv-1, curr_t_temp.expand(x_adv.shape[0]))
                    if args.domain=='imagenet':
                        score, _ = torch.split(score, score.shape[1]//2, dim=1)
                        assert x_adv.shape == score.shape, f'{x_adv.shape}, {score.shape}'
                    if args.detection_ensattack_norm_flag:
                        score_adv_list.append(score.detach().view(x_adv.shape[0],-1).norm(dim=-1).unsqueeze(0))
                    elif args.single_vector_norm_flag:
                        # score_adv_list.append(score.detach().view(1, *x_adv.shape))
                        score_sum += score.detach()
                    else:
                        score_adv_list.append(score.detach())
                
                if args.generate_lw_flag:
                    if args.detection_ensattack_norm_flag:
                        exit(0)
                    elif args.single_vector_norm_flag:
                        if not args.clean_score_flag:
                            path_1w = f'{args.detection_datapath}/scores_adv_{attack_method}_{args.epsilon}_{args.num_steps}single_vector_norm{value}{isperb_image}{data_size}'
                        else:
                            path_1w = f'{args.detection_datapath}/scores_clean{isnorm}single_vector_norm{value}{isperb_image}{data_size}'
                        os.makedirs(path_1w, exist_ok=True)
                        for n in range(score_sum.size(0)):
                            np.save(path_1w +"/{}.npy".format(j+n+1), ((score_sum[n])/value).cpu().numpy())
                    else:
                        exit(0)
                    j += (n+1)
                
                else:
                    if args.detection_ensattack_norm_flag:
                        score_adv_lists.append(torch.cat(score_adv_list, dim=0))
                    elif args.single_vector_norm_flag:
                        score_adv_lists.append(score_sum/value)
                    else:
                        score_adv_lists.append(torch.cat(score_adv_list, dim=0).view(len(score_adv_list),*x_adv.shape).cpu())
                    score_adv_list.clear()
        
        if not args.generate_lw_flag:
            print(f'attack_method: {attack_method}, robust accuracy: top1:{top1_counter/num_samples}--top5:{top5_counter/num_samples}')
            print(f'attack and diffuison time: {time.time() - start_time}')
            score_tensor = torch.cat(score_adv_lists, dim=1) if not args.single_vector_norm_flag else torch.cat(score_adv_lists, dim=0)
            print(f"score_tensor.shape:{score_tensor.shape}")
            isnorm = '_norm' if args.detection_ensattack_norm_flag else ''
            isperb_image = 'perb_image' if args.perb_image else ''
            data_size = '' if args.num_sub==500 else str(args.num_sub)
            if not args.clean_score_flag:
                if not args.single_vector_norm_flag:
                    np.save(f'{args.detection_datapath}/scores_adv_{attack_method}_{args.epsilon}_{args.num_steps}{isnorm}{value}{isperb_image}{data_size}.npy', score_tensor.data.cpu().numpy())
                else:
                    np.save(f'{args.detection_datapath}/scores_adv_{attack_method}_{args.epsilon}_{args.num_steps}single_vector_norm{value}{isperb_image}{data_size}.npy', score_tensor.data.cpu().numpy())
            
            else:
                if not args.single_vector_norm_flag:
                    np.save(f'{args.detection_datapath}/scores_clean{isnorm}{value}{isperb_image}{data_size}.npy', score_tensor.data.cpu().numpy())
                else:
                    np.save(f'{args.detection_datapath}/scores_clean{isnorm}single_vector_norm{value}{isperb_image}{data_size}.npy', score_tensor.data.cpu().numpy())
            score_adv_lists.clear()
            
  

if __name__ == '__main__':
    args, config = parse_args_and_config()
    robustness_eval(args, config)
    