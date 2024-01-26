import numpy as np
from skimage.metrics import structural_similarity
import lpips
import torch
from pytorch_fid_master.src.pytorch_fid.fid_score import calculate_frechet_distance, calculate_activation_statistics
from pytorch_fid_master.src.pytorch_fid.inception import InceptionV3

def metrics_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    if max_value == 1:
        img1 = (img1 - np.min(img1))/(np.max(img1) - np.min(img1))
        img1 = (img2 - np.min(img2))/(np.max(img2) - np.min(img2))
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def metrics_ssim(img1, img2):
    data_range = max(np.max(img1), np.max(img2)) - min(np.min(img1), np.min(img2))
    # data_range = np.max(img1) - np.min(img1)
    # data_range = np.max(img2) - np.min(img2)
    (score, diff) = structural_similarity(img1, img2, full=True, data_range=data_range)
    return score

def metrics_lpips(img1, img2): 
    img1 = torch.tensor(img1, dtype=torch.float32)
    img2 = torch.tensor(img2, dtype=torch.float32)
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    score = loss_fn_alex(img1, img2)
    return score.item()

def metrics_fid(img1, img2):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    batch_size=50
    dims=2048
    num_workers=1
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device).float()
    m1, s1 = calculate_activation_statistics([img1], model, batch_size=batch_size, dims=dims,
                                    device=device, num_workers=num_workers)
    m2, s2 = calculate_activation_statistics([img2], model, batch_size=batch_size, dims=dims,
                                    device=device, num_workers=num_workers)
    score = calculate_frechet_distance(m1, s1, m2, s2)
    return score