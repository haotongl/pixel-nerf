import argparse, os
import numpy as np
from PIL import Image
import re
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import lpips
import cv2
import torch
lpips_func = lpips.LPIPS(net='vgg')

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def read_gt_images(scene_name, halfres):
    if 'scan' in scene_name:
        splits = np.load(os.path.join('data/dtu_rect/Splits', scene_name,'split.npy'), allow_pickle=True).item()
        test_set = splits['test']
        image_path_pattern = 'data/dtu_rect/Rectified/{}_train/rect_{:03d}_3_r5000.png'
        images = [np.array(Image.open(image_path_pattern.format(scene_name, i+1))) for i in test_set]
        images = (np.stack(images) / 255.).astype(np.float32)

        dpt_pattern = 'data/dtu_rect/Depths_Rect/{}/depth_map_{:04d}.pfm'
        depths = [read_pfm(dpt_pattern.format(scene_name, i))[0] for i in test_set]
        depths = np.stack(depths)
        masks = np.logical_and(depths>425., depths<935.)
    else:
        __import__('ipdb').set_trace()

    if halfres:
        images = [cv2.resize(images[i], None, fx=0.5, fy=0.5) for i in range(len(images))]
        images = np.stack(images).astype(np.float32)
        if masks is not None:
            masks = [cv2.resize(masks[i].astype(np.uint8), None, fx=0.5, fy=0.5) for i in range(len(masks))]
            masks = np.stack(masks).astype(np.bool_)
    return images, masks

# def read_gt_images(scene_name):
#     if 'scan' in scene_name:
#         splits = np.load(os.path.join('data/dtu_rect/Splits', scene_name,'split.npy'), allow_pickle=True).item()
#         test_set = splits['test']
#         image_path_pattern = 'rs_dtu_4/DTU/{}/image/{:06d}.png'
#         images = [np.array(Image.open(image_path_pattern.format(scene_name, i))) for i in test_set]
#         images = (np.stack(images) / 255.).astype(np.float32)

#         dpt_pattern = 'data/dtu_rect/Depths_Rect/{}/depth_map_{:04d}.pfm'
#         depths = [read_pfm(dpt_pattern.format(scene_name, i))[0] for i in test_set]
#         depths = np.stack(depths)
#         masks = np.logical_and(depths>425., depths<935.)
#     else:
#         __import__('ipdb').set_trace()

#     return images, masks
def read_dtu_pred_images(scene_path):
    images = os.listdir(scene_path)
    images.sort()
    preds = []
    gts = []
    for image in images:
        if 'compare' not in image:
            continue
        img = np.array(Image.open(scene_path + '/' + image))
        pred = img[:, :320]
        gt = img[:, 320:]
        preds.append(pred/255.)
        gts.append(gt/255.)
    return np.stack(gts), np.stack(preds)

def run_main(args):
    root_name = args.evaldir.split('/')[-1].split('_')[0]
    ps, ss, ls = [], [], []
    for en in os.scandir(args.evaldir):
        if not en.is_dir():
            continue
        gt_images, gt_masks = read_gt_images(en.name, True)
        # gt_masks = None
        gt_images, pred_images = read_dtu_pred_images(en.path)
        p, s, l = eval_metric(en.name, gt_images, pred_images, gt_masks)
        ps.append(p)
        ss.append(s)
        ls.append(l)
    print('mean: ', np.mean(ps), np.mean(ss), np.mean(ls))
    # image_names = os.listdir(args.imgdir)
    # image_names.sort()
    # images = [np.array(Image.open(args.imgdir + '/' + image_name)) for image_name in image_names]
    # pred_images = (np.stack(images) / 255.)


def eval_metric(expname, gt_images, pred_images, gt_masks):
    metric_psnrs, metric_ssims, metric_lpips = [], [], []
    for i in range(len(gt_images)):
        if gt_masks is not None:
            metric_psnrs.append(psnr(gt_images[i][gt_masks[i]], pred_images[i][gt_masks[i]], data_range=1.))
            gt_images[i][gt_masks[i]==False] = 0.
            pred_images[i][gt_masks[i]==False] = 0.
            gt, pred = torch.Tensor(gt_images[i])[None].permute(0, 3, 1, 2), torch.Tensor(pred_images[i])[None].permute(0, 3, 1, 2)
            gt, pred = (gt-0.5)*2., (pred-0.5)*2.
            lpips_item = lpips_func(gt, pred).item()
            metric_lpips.append(lpips_item)
            metric_ssims.append(ssim(gt_images[i], pred_images[i], multichannel=True, data_range=1.))
        else:
            metric_psnrs.append(psnr(gt_images[i], pred_images[i], data_range=1.))
            metric_ssims.append(ssim(gt_images[i], pred_images[i], multichannel=True, data_range=1.))
            gt, pred = torch.Tensor(gt_images[i])[None].permute(0, 3, 1, 2), torch.Tensor(pred_images[i])[None].permute(0, 3, 1, 2)
            gt, pred = (gt-0.5)*2., (pred-0.5)*2.
            lpips_item = lpips_func(gt, pred).item()
            metric_lpips.append(lpips_item)
    print(expname, 'psnr: {:.3f} ssim: {:.3f} lpips: {:.4f}'.format(np.mean(metric_psnrs), np.mean(metric_ssims), np.mean(metric_lpips)))
    return np.mean(metric_psnrs), np.mean(metric_ssims), np.mean(metric_lpips)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='main')
    parser.add_argument('--evaldir', type=str, default='logs/scan114_test')
    parser.add_argument('--imgdir', type=str, default='logs/scan114_test/testset_200000')
    parser.add_argument('--halfres', type=bool, default=False)
    args = parser.parse_args()
    globals()['run_' + args.type](args)
