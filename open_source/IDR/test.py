import argparse
import pickle

import torch.nn.functional as F
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse

from utils import *
# from models import UNet_n2n_un
from models import build_model


def model_forward(net, noisy, padding=32):
    h, w, _ = noisy.shape
    pw, ph = (w + 31) // 32 * 32 - w, (h + 31) // 32 * 32 - h
    with torch.no_grad():
        input_var = torch.FloatTensor([noisy]).cuda().permute(0, 3, 1, 2)
        input_var = F.pad(input_var, (0, pw, 0, ph), mode='reflect')
        # print(input_var.shape,  noisy.shape, ph, pw)
        out_var = net(input_var)

    if pw != 0:
        out_var = out_var[..., :, :-pw]
    if ph != 0:
        out_var = out_var[..., :-ph, :]

    denoised = out_var.permute([0, 2, 3, 1])[0].detach().cpu().numpy()
    return denoised


def add_noise(clean, ntype, sigma=None):
    # assert ntype.lower() in ['gaussian', 'gaussian_gray', 'impulse', 'binomial', 'pattern1', 'pattern2', 'pattern3', 'line']
    assert sigma < 1
    if 'gaussian' in ntype:
        noisy = clean + np.random.normal(0, sigma, clean.shape)

    elif ntype == 'binomial':
        h, w, c = clean.shape
        mask = np.random.binomial(n=1, p=(1 - sigma), size=(h, w, 1))
        noisy = clean * mask

    elif ntype == 'impulse':
        mask = np.random.binomial(n=1, p=(1 - sigma), size=clean.shape)
        noisy = clean * mask

    elif ntype[:4] == 'line':
        # sigma = 25 / 255.0
        h, w, c = clean.shape
        line_noise = np.ones_like(clean) * np.random.normal(0, sigma, (h, 1, 1))
        noisy = clean + line_noise

    elif ntype[:7] == 'pattern':
        # sigma = 5 / 255.0
        h, w, c = clean.shape
        n_type = int(ntype[7:])

        one_image_noise, _, _ = get_experiment_noise('g%d' % n_type, sigma, 0, (h, w, 3))
        noisy = clean + one_image_noise
    else:
        assert 'not support %s' % args.ntype

    return noisy


def load_npz(npz_path, key):
    data = np.load(npz_path)[key]
    split_train_test_path = os.path.join(os.path.dirname(npz_path), 'Train2TrainValTestv2.pkl')
    if os.path.exists(split_train_test_path):
        with open(split_train_test_path, 'rb') as f:
            train_test_idx = pickle.load(f)
        data = data[train_test_idx['test_idx']]
        print('test split used')
    # unstack z plane
    images = np.transpose(data, (0, 2, 1, 3, 4))
    images = np.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
    # reshape to n x h x w x c (necessary dimensions for IDR)
    n, c, h, w = images.shape
    images = np.reshape(images, (n, h, w, c))
    print(f'Shape of the dataset: {images.shape}')
    return images


def load_noisy_clean_pairs(name):
    print(f'dataset: {name}')

    if name == 'planaria_2D':
        npz_path = '/home/schaudhary/siva_projects/denoising/data/Denoising_Planaria/data_label.npz'
    elif name == 'tribolium_2D':
        npz_path = '/home/schaudhary/siva_projects/denoising/data/Denoising_Tribolium/train_data/data_label.npz'
    elif name == 'flywing':
        npz_path = '/home/schaudhary/siva_projects/denoising/data/Projection_Flywing/train_data/data_label.npz'
    else:
        raise ValueError(f'Unsupported data name {name}')
    
    images_noisy = load_npz(npz_path, 'X')
    images_clean = load_npz(npz_path, 'Y')
    return images_noisy, images_clean


def rescale(im):
    min = im.min()
    max = im.max()
    return (im-min) / (max-min)


def test(args, net, dataset_name):
    if args.save_img:
        save_dir = os.path.join(args.res_dir, dataset_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    images_noisy, images_clean = load_noisy_clean_pairs(dataset_name)
    total_images = images_noisy.shape[0]
    res = {'psnr': [], 'ssim': [], 'mse': []}
    for idx, (noisy, gt) in enumerate(zip(images_noisy, images_clean)):
        #if idx > 1000: break # for debugging
        noisy = rescale(noisy)
        gt = rescale(gt)

        if args.zero_mean:
            noisy = noisy - 0.5

        print('\rprocessing ', idx, '/', total_images, end='')
        denoised = model_forward(net, noisy)

        denoised = denoised + (0.5 if args.zero_mean else 0)
        denoised = np.clip(denoised, 0, 1)
        noisy = noisy + (0.5 if args.zero_mean else 0)

        # save PSNR
        temp_psnr = compare_psnr(gt, denoised)
        temp_ssim = compare_ssim(gt, denoised, multichannel=True)
        temp_mse = compare_mse(gt, denoised)

        res['psnr'].append(temp_psnr)
        res['ssim'].append(temp_ssim)
        res['mse'].append(temp_mse)

        if args.save_img:
            filename = str(idx)
            # convert to uint8 for imwrite
            denoised = np.clip(denoised * 255.0 + 0.5, 0, 255).astype(np.uint8)
            noisy = np.clip(noisy * 255.0 + 0.5, 0, 255).astype(np.uint8)
            gt = np.clip(gt * 255.0 + 0.5, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, '%s_%.2f_out.png' % (filename, temp_psnr)), denoised)
            cv2.imwrite(os.path.join(save_dir, '%s_NOISY.png' % (filename)), noisy)
            cv2.imwrite(os.path.join(save_dir, '%s_GT.png' % (filename)), gt)

    print('\nssim %.3f | mse %.3f | psnr %.2f' % (np.mean(res['ssim']), np.mean(res['mse']), np.mean(res['psnr'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='self supervised')
    parser.add_argument('--dataset_name', default=None, type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--res_dir', default="test_output", type=str)
    parser.add_argument('--save_img', default=True, type=bool)
    args = parser.parse_args()

    args.zero_mean = False
    if args.model_path is None:
        args.model_path = 'checkpoint/%s.pth' % args.ntype

    print('Testing', args.model_path)

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    # model
    ch = 1 # grayscale images
    cfg = EasyDict()
    cfg.model_name = 'UNet_n2n_un'
    cfg.model_args = {'in_channels': ch, 'out_channels': ch}
    net = build_model(cfg)

    net = net.cuda()
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    test(args, net, args.dataset_name)
