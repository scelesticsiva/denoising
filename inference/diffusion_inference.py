import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import glob
import time
import torchvision as thv

from math import ceil
from omegaconf import OmegaConf
from sampler import BaseSampler
from utils import util_common
from utils import util_image
from torch.utils import data
from torch.nn.functional import pad
from torch.profiler import profile, record_function, ProfilerActivity

from inference.data import npz_dataset
from inference.val_utils import filter_val_files

class DiffusionSampler(BaseSampler):
    def sample_func(self, noise=None, start_timesteps=None, bs=4, num_images=1000):
        print('Begining sampling:')

        h = w = self.configs.im_size
        if self.num_gpus != 0:
            total_iters = ceil(num_images / (bs * self.num_gpus))
        else:
            total_iters = ceil(num_images / (bs * 1))
        for ii in range(total_iters):
            if self.rank == 0 and self.display:
                print(f'Processing: {ii+1}/{total_iters}')
            if noise == None:
                if self.num_gpus != 0:
                    noise = torch.randn((bs, 1, h, w), dtype=torch.float32).cuda()
                else:
                    noise = torch.randn((bs, 1, h, w), dtype=torch.float32)
            if 'ddim' in self.configs.diffusion.params.timestep_respacing:
                sample = self.diffusion.ddim_sample_loop(
                        self.model,
                        shape=(bs, 1, h, w),
                        noise=noise,
                        start_timesteps=start_timesteps,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None,
                        device=None,
                        progress=False,
                        eta=0.0,
                        )
            else:
                sample = self.diffusion.p_sample_loop(
                        self.model,
                        shape=(bs, 1, h, w),
                        noise=noise,
                        start_timesteps=start_timesteps,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None,
                        device=None,
                        progress=False,
                        )
            sample = util_image.normalize_th(sample, reverse=True).clamp(0.0, 1.0)
        return sample

def collate_fn(data):
    # assume data is a list of dicts with 2 keys: {img, fpth}
    # pad all images to same number of channels
    imgs = [torch.from_numpy(d['img']) for d in data]
    orig_channels = [img.shape[0] for img in imgs]
    channel_padding = max(orig_channels)
    imgs = [pad(img, ((0, 0, 0, 0, 0, channel_padding-img.shape[0])), 'constant', 0) for img in imgs]
    imgs = torch.stack(imgs) 
    fpths = [d['fpth'] for d in data]
    return {'img': imgs, 'fpth': fpths, 'orig_channels': orig_channels}

def main(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)
    transform = thv.transforms.Normalize(mean=(0.5), std=(0.5))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    timestep_respacing = '1000'

    configs = OmegaConf.load(cfg.cfg_path)
    configs.gpu_id = cfg.gpu_id
    configs.diffusion.params.timestep_respacing = timestep_respacing

    sampler_dist = DiffusionSampler(configs)

    tN = torch.tensor(cfg.tN).to(device)

    all_val_files = filter_val_files(cfg.noisy_img_dir, val_files=cfg.val_files, ext='*.npz')
    dataset = npz_dataset.NPZ3DDataset(all_val_files, key='pred')
    dataloader = data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=cfg.num_workers)
    print(f'Device availability:{device}')

    for batch in dataloader:
        st = time.time()
        img = batch['img'].to(device)
        orig_bsz, channels, H, W = img.shape
        # stack all images into one batch
        img = img.view(orig_bsz * channels, 1, H, W)
        y0 = transform(img)
        xN_given_y0 = sampler_dist.diffusion.q_sample(y0, tN)
        # NOTE(matt) - the bottleneck is somewhere in sample_func;
        # but the profiler is not correctly capturing CUDA calls.
        # may need to update pytorch to get the profiler working.
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("sample_func"):
                x0_sampled = sampler_dist.sample_func(
                    noise=xN_given_y0,
                    start_timesteps=tN,
                    bs=len(y0),
                    num_images=len(y0),
                )
        prof_stats = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        print(prof_stats)
        with open(cfg.save_dir, 'prof_stats.txt', 'w') as f:
            f.write(prof_stats)
        #print(f'time (sample): {time.time() - st:.4f} seconds')
        #print(x0_sampled.shape)
        # unstack output
        x0_sampled = x0_sampled.view(orig_bsz, channels, H, W)
        # Save the denoised
        for i in range(orig_bsz):
            save_fpth = os.path.join(cfg.save_dir, os.path.basename(batch['fpth'][i]))
            # un-pad channel dim if necessary
            orig_channels = batch['orig_channels'][i]
            to_save = x0_sampled[i, :orig_channels, ...].cpu().numpy()
            print(to_save.shape)
            np.savez_compressed(save_fpth, pred=to_save)
            print(f'Saved pred to {save_fpth}, shape of pred: {to_save.shape}, time: {time.time() - st:.4f} seconds')

if __name__ == '__main__':
    from munch import Munch

    config = Munch.fromDict(dict(
        cfg_path='/home/sivark/repos/denoising/open_source/DifFace/configs/sample/iddpm_planaria_LargeMdl.yaml',
        gpu_id='0',
        #tN=500,
        tN=50,
        batch_size=4,
        num_workers=4,
        noisy_img_dir='/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/n2v_pred/condition_1/',
        #save_dir='/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/diffusion_pred/t500_from_n2v/condition_1/',
        save_dir='/home/mds/data/denoising/test/',
        val_files=['EXP280_Smed_live_RedDot1_slide_mnt_N3_stk1',
                   'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0003',
                   'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0005',
                   'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0013',
                   'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0009'],
    ))

    main(config)



