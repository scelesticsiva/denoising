
from csbdeep.models import Config, CARE
from metrics import metrics_psnr, metrics_ssim, metrics_lpips, metrics_fid

import glob
import numpy as np
import pandas as pd
import tifffile


def collate_noisy_gt(noisy_paths, gt_paths):
    data = {X:Y for X, Y in zip(noisy_paths, gt_paths)}
    return data

def load_planaria_test_data():
    test_imgs_X1 = glob.glob('/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/CARE-dataset/Denoising_Planaria/test_data/condition_1/*.tif')
    test_imgs_X2 = glob.glob('/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/CARE-dataset/Denoising_Planaria/test_data/condition_2/*.tif')
    test_imgs_X3 = glob.glob('/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/CARE-dataset/Denoising_Planaria/test_data/condition_3/*.tif')
    test_imgs_Y = glob.glob('/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/CARE-dataset/Denoising_Planaria/test_data/GT/*.tif')
    data = {}
    data.update(collate_noisy_gt(test_imgs_X1, test_imgs_Y))
    data.update(collate_noisy_gt(test_imgs_X2, test_imgs_Y))
    data.update(collate_noisy_gt(test_imgs_X3, test_imgs_Y))
    return data

if __name__ == '__main__':
    
    data = load_planaria_test_data()

    axes = 'YX'
    model_name = 'care_planaria_2D'
    model_basedir = '/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/baseline_denoising/care'
    model = CARE(config=None, name=model_name, basedir=model_basedir)

    metrics_scores = []
    for noisy in data:
        X = tifffile.imread(noisy).astype(np.float32)
        Y = tifffile.imread(data[noisy]).astype(np.float32)
        assert X.ndim == Y.ndim
        # if X.ndim >= 3:
        #     res = np.reshape()
        for z in range(X.shape[0]):
            restored = model.predict(X[z], axes)
            psnr_score = metrics_psnr(img1=restored, img2=Y[z], max_value=1)
            ssim_score = metrics_ssim(img1=restored, img2=Y[z])
            lpips_score = metrics_lpips(img1=restored, img2=Y[z])
            fid_score = metrics_fid(img1=restored, img2=Y[z])
            metrics_scores.append([z, noisy, data[noisy], model_name, model_basedir, psnr_score, ssim_score, lpips_score, fid_score])
    df = pd.DataFrame(metrics_scores, columns=['z', 'noisy_path', 'gt_path', 'model', 'basedir', 'psnr', 'ssim', 'lpips', 'fid'])
    
    out_path = f'/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/baseline_denoising/accuracy_{model_name}.pickle'
    df.to_pickle(out_path)

    


