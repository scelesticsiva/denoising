import json
import glob
import os
import tifffile
import numpy as np

from munch import Munch
from PIL import Image
from n2v.models import N2VConfig, N2V

from functools import partial
from multiprocessing import Pool

def predict_single_vol(fpth, model, pred_save_dir):
    fname = os.path.basename(fpth)
    dest_fpth = os.path.join(pred_save_dir, fname)

    curr_img = np.load(fpth, allow_pickle=True)['img']
    curr_vol = np.expand_dims(curr_img, axis=-1)
    all_preds = []
    for curr_slice in curr_vol:
        curr_pred = model.predict(curr_slice, axes='YXC')
        all_preds.append(curr_pred[..., 0][None, ...])

    np.savez_compressed(dest_fpth, pred=np.concatenate(all_preds, axis=0))
    print(f'Saving predictions to {dest_fpth}')

def read_tif_as_npy(tif_fpth):
    img = tifffile.imread(tif_fpth).astype(np.float32)
    return img

def infer(config):
    model = N2V(config=None, name=config.model_name, basedir=config.basedir)
    model.load_weights(name=config.weights_name)

    os.makedirs(config.pred_save_dir, exist_ok=True)

    all_npz = glob.glob(config.test_data_dir + '*.npz')
    all_npz = [e for e in all_npz if not os.path.exists(os.path.join(config.pred_save_dir, os.path.basename(e)))]
    print(f'Total number of noisy images found in destination: {len(all_npz)}')

    partial_obj = partial(predict_single_vol, model=model, pred_save_dir=config.pred_save_dir)
    # pool_obj = Pool(4)
    # pool_obj.map(partial_obj, all_npz)
    # pool_obj.close()

    for fpth in all_npz:
        partial_obj(fpth)


if __name__ == '__main__':
    # config_fpth = '/home/scelesticsiva/Documents/projects_w_shivesh/denoising/baseline_denoising/N2V/N2V_planaria/config.json'
    # model_name = 'planaria'
    # basedir = 'inference_assets'
    # weights_pth = '/home/scelesticsiva/Documents/projects_w_shivesh/denoising/baseline_denoising/N2V/N2V_planaria/weights_best.h5'

    for condition_name in ['condition_1', 'condition_2', 'condition_3']:
        infer_config = Munch.fromDict(dict(
            model_name='N2V_planaria',
            basedir='/home/sivark/supporting_files/denoising/trained_models/n2v/planaria/',
            test_data_dir=f'/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/test_data_preprocessed/{condition_name}/',
            weights_name='weights_best.h5',
            pred_save_dir=f'/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/n2v_pred/{condition_name}/',
        ))

        infer(infer_config)