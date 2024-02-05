from math import ceil
from sampler import BaseSampler
from utils import util_image
import torch
import torchvision as thv
import numpy as np

class DiffusionSampler(BaseSampler):
    def sample_func(self, noise=None, start_timesteps=None, bs=4, num_images=1000, save_dir=None):
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
    
    def repaint_style_sample(self, start_input, tN=100, repeat_timestep=1, num_repeats=1, mixing=False, mixing_stop=50):
        print('Begining sampling:')
        
        transform = thv.transforms.Normalize(mean=(0.5), std=(0.5))
        start_input = transform(start_input)
        h = w = self.configs.im_size
        bs = start_input.shape[0]
        shape = (bs, 1, h, w)
        if start_input != None:
            img = self.diffusion.q_sample(start_input, torch.tensor(tN, device=start_input.device))
        else:
            img = torch.randn((bs, 1, h, w))

        while tN > 0:
            indices = list(range(tN)[::-1])[0:repeat_timestep]
            print(f'repainting between {indices[0]}-{indices[-1]} for {num_repeats} times')
            for rp in range(num_repeats):
                # img = self.diffusion.q_sample(img, indices[0])
                for i in indices:
                    t = torch.tensor([i] * shape[0], device=img.device)
                    with torch.no_grad():
                        out = self.diffusion.p_sample(
                            self.model,
                            img,
                            t,
                            clip_denoised=True,
                            denoised_fn=None,
                            model_kwargs=None,
                        )
                        img = out["sample"]
                    if mixing == True and i > mixing_stop:
                        xN_given_y0 = self.diffusion.q_sample(start_input, torch.tensor(i, device=start_input.device))
                        img = 0.5*img + 0.5*xN_given_y0
            tN -= repeat_timestep
        sample = util_image.normalize_th(img, reverse=True).clamp(0.0, 1.0)
        return sample
    
    
    def mixing_sample(self, start_input, mixing_input, tN=100, mix_start=500, mix_stop=100, n_steps=400, alpha=0.1):
        def linear_mixing_alphas(min_alpha, max_alpha, n_steps):
            mixing_alphas = np.linspace(min_alpha, max_alpha, n_steps - 1)
            return mixing_alphas

        def constant_mixing_alphas(alpha, n_steps):
            mixing_alphas = [alpha for i in range(n_steps - 1)]
            return mixing_alphas

        transform = thv.transforms.Normalize(mean=(0.5), std=(0.5))
        start_input = transform(start_input)
        mixing_input = transform(mixing_input)

        assert (mix_start - mix_stop) % n_steps == 0
        mixing_t_steps = {mix_stop + i*int((mix_start - mix_stop)/n_steps):i for i in range(1, n_steps)}
        mixing_alphas = constant_mixing_alphas(alpha, n_steps)
        # mixing_alphas = linear_mixing_alphas(0.01, 0.99, n_steps)
        # TODO calculate mixing_xN_given_y0 inside mixing loop
        mixing_xN_given_y0 = [self.diffusion.q_sample(mixing_input, torch.tensor(step)) for step in mixing_t_steps]
        xN_given_y0 = self.diffusion.q_sample(start_input, tN)
        img = xN_given_y0

        all_imgs = []
        all_imgs.append(img)
        h = w = self.configs.im_size
        bs = img.shape[0]
        shape = (bs, 1, h, w)
        indices = list(range(tN)[::-1])
        for i in indices:
            t = torch.tensor([i] * shape[0], device=start_input.device)
            with torch.no_grad():
                out = self.diffusion.p_sample(
                    self.model,
                    img,
                    t,
                    clip_denoised=True,
                    denoised_fn=None,
                    model_kwargs=None,
                )
                img = out["sample"]
                all_imgs.append(img)
            if i in mixing_t_steps:
                print(f'mixing at {i}', (1 - mixing_alphas[mixing_t_steps[i] - 1]), mixing_alphas[mixing_t_steps[i] - 1])
                img = (1 - mixing_alphas[mixing_t_steps[i] - 1])*img + mixing_alphas[mixing_t_steps[i] - 1]*mixing_xN_given_y0[mixing_t_steps[i] - 1]
            sample = util_image.normalize_th(img, reverse=True).clamp(0.0, 1.0)
        return sample
    
    def diffusion_denoise(self, inputs, X_shape, sampler_fn, n_samples):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        y0 = []
        for X in inputs:
            if X_shape == 'BZHWC':
                B, Z, H, W, C = X.shape
                X = np.reshape(X, [-1, H, W, C])
                X = X[:, :, :, 0]
            elif X_shape == 'BZHW':
                B, Z, H, W = X.shape
                X = np.reshape(X, [-1, H, W])
            elif X_shape == 'BHWC':
                B, H, W, C = X.shape
                X = X[:, :, :, 0]
            elif X_shape == 'BHW':
                X = X
            else:
                raise ValueError(f'invalid X_shape {X_shape}')
            curr_y0 = torch.tensor(X, dtype=torch.float32, device=device)
            curr_y0 = curr_y0[:, None, :, :]
            y0.append(curr_y0)

        all_samples = []
        for n in range(n_samples):
            x0_sampled = sampler_fn(*y0)
            all_samples.append(x0_sampled)
        x0_sampled = torch.mean(torch.stack(all_samples, dim=0), dim=0)


        if X_shape == 'BZHWC':
            x0_sampled = np.reshape(x0_sampled.cpu().numpy(), [B, Z, H, W])
        elif X_shape == 'BZHW':
            x0_sampled = np.reshape(x0_sampled.cpu().numpy(), [B, Z, H, W])
        elif X_shape == 'BHWC':
            x0_sampled = x0_sampled.cpu().numpy()[:, 0, :, :]
        elif X_shape == 'BHW':
            x0_sampled = x0_sampled.cpu().numpy()[:, 0, :, :]
        return x0_sampled