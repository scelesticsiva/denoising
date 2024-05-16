import numpy as np
from utils.option import parse
import torch
import os
import pickle
import tifffile


def split_train_test(data, npz_path, split):
    split_train_test_path = os.path.join(os.path.dirname(npz_path), 'Train2TrainValTestv2.pkl')
    if os.path.exists(split_train_test_path):
        with open(split_train_test_path, 'rb') as f:
            train_test_idx = pickle.load(f)
        data = data[train_test_idx[split]]
        print(f'{split} split used')
    return data

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    base_path = '/home/schaudhary/siva_projects/SpatiallyAdaptiveSSID' 

    # # planaria
    # opt_path = os.path.join(base_path, 'option', 'train_care_planaria_2D.json')
    # model_save_path = os.path.join(base_path, 'logs', 'train_care_planaria_2D_3', 'model_iter_00300000.pth')
    # npz_path = '/home/schaudhary/siva_projects/denoising/data/Denoising_Planaria/data_label.npz'
    # ssid_pred_save_path = os.path.join(os.path.dirname(npz_path), 'ssid_pred_test_2.npz')

    # # tribolium
    # opt_path = os.path.join(base_path, 'option', 'train_care_tribolium_2D.json')
    # model_save_path = os.path.join(base_path, 'logs', 'train_care_tribolium_2D_3', 'model_iter_00300000.pth')
    # npz_path = '/home/schaudhary/siva_projects/denoising/data/Denoising_Tribolium/train_data/data_label.npz'
    # ssid_pred_save_path = os.path.join(os.path.dirname(npz_path), 'ssid_pred_test_2.npz')
    

    # flywing
    opt_path = os.path.join(base_path, 'option', 'train_care_flywing.json')
    model_save_path = os.path.join(base_path, 'logs', 'train_care_flywing_2', 'model_iter_00900000.pth')
    npz_path = '/home/schaudhary/siva_projects/denoising/data/Projection_Flywing/train_data/data_label.npz'
    ssid_pred_save_path = os.path.join(os.path.dirname(npz_path), 'ssid_pred_test_2.npz')
    
    
    opt = parse(opt_path)
    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()
    model.load_model(model_save_path, device)
    model.networks['UNet'].eval()

    X = np.load(npz_path)['X']
    X = split_train_test(X, npz_path, 'test_idx')
    Y = np.load(npz_path)['Y']
    Y = split_train_test(Y, npz_path, 'test_idx')
    print(X.shape, Y.shape)


    # # save indiviual
    # idx = 2
    # input = X[idx]
    # input = np.transpose(input, (1, 0, 2, 3))
    # input = np.repeat(input, 3, axis=1)
    # input = torch.tensor(input, device=device)
    # print('input', input.shape)
    # model.networks['UNet'].eval()
    # with torch.no_grad():
    #     output = model.networks['UNet'](input)
    # print('output', output.shape)

    # save_X = np.max(input[:, 1, :, :].cpu().numpy(), axis=0)
    # save_pred = np.max(output[:, 1, :, :].cpu().numpy(), axis=0)
    # save_Y = np.max(Y[idx, 0], axis=0)
    # print(save_X.shape)
    # print(save_pred.shape)
    # print(save_Y.shape)
    # tifffile.imwrite('dummy_X.tif', save_X)
    # tifffile.imwrite('dummy_pred.tif', save_pred)
    # tifffile.imwrite('dummy_Y.tif', save_Y)


    ssid_pred = []
    for idx in range(X.shape[0]):
        input = X[idx]
        input = np.transpose(input, (1, 0, 2, 3))
        input = np.repeat(input, 3, axis=1)
        input = torch.tensor(input, device=device)
        with torch.no_grad():
            output = model.networks['UNet'](input)
        ssid_pred.append(output[:, 0, :, :].cpu().numpy())
    ssid_pred = np.stack(ssid_pred, axis=0)
    print(ssid_pred.shape)
    print(f'saved at - {ssid_pred_save_path}')
    np.savez(ssid_pred_save_path, ssid_pred=ssid_pred)
    
    



