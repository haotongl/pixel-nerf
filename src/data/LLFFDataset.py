import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2
from util import get_image_to_tensor_balanced, get_mask_to_tensor


class LLFFDataset(torch.utils.data.Dataset):
    """
    Dataset from DVR (Niemeyer et al. 2020)
    Provides 3D-R2N2 and NMR renderings
    """

    def __init__(
        self,
        path,
        stage="train",
        list_prefix="softras_",
        image_size=None,
        sub_format="shapenet",
        scale_focal=True,
        max_imgs=100000,
        z_near=1.2,
        z_far=4.0,
        skip_step=None,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
        :param sub_format shapenet | dtu dataset sub-type.
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()
        self.base_path = path
        assert os.path.exists(self.base_path)

        cats = [x for x in glob.glob(os.path.join(path, "*")) if os.path.isdir(x)]

        sub_format = "dtu"
        if stage == "train":
            file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        elif stage == "val":
            file_lists = [os.path.join(x, list_prefix + "val.lst") for x in cats]
        elif stage == "test":
            file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]

        all_objs = [ [path, cat] for cat in cats ]
        self.all_objs = all_objs
        self.stage = stage

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        print(
            "Loading DVR dataset",
            self.base_path,
            "stage",
            stage,
            len(self.all_objs),
            "objs",
            "type:",
            sub_format,
        )
        self.image_size = image_size
        if sub_format == "dtu":
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        self.sub_format = sub_format
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False

    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        cat, root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob.glob(os.path.join(root_dir, "images_4", "*"))
            if (x.endswith(".jpg") or x.endswith(".png") or x.endswith('.JPG'))
        ]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob.glob(os.path.join(root_dir, "mask", "*.png")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]
            mask_paths = [mask_paths[i] for i in sel_indices]

        cam_path = os.path.join(root_dir, "scene_meta_.npy")
        all_cam = np.load(cam_path, allow_pickle=True).item()

        for rgb_path, image_name in zip(rgb_paths, all_cam['image_names']):
            assert(image_name == rgb_path.split('/')[-1])
        # all_poses = [np.linalg.inv(all_cam['poses'][i]) for i in range(len(all_cam['poses']))]
        # cam_to_world
        all_poses_ = [all_cam['poses'][i] for i in range(len(all_cam['poses']))]
        all_poses = [self._coord_trans_world @ torch.tensor(all_cam['poses'][i], dtype=torch.float32) @ self._coord_trans_cam for i in range(len(all_cam['poses']))]
        # all_poses_ = [torch.tensor(all_cam['poses'][i], dtype=torch.float32) for i in range(len(all_cam['poses']))]
        # all_poses = [torch.cat([pose[:, :1], -pose[:, 1:2], -pose[:, 2:3], pose[:, 3:]], dim=-1) for pose in all_poses_]
        np_imgs = [imageio.imread(rgb_path) for rgb_path in rgb_paths]
        H, W = np_imgs[0].shape[:2]
        tar_h, tar_w = 640, 960
        np_imgs = [cv2.resize(img, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR) for img in np_imgs]
        all_imgs = [self.image_to_tensor(img) for img in np_imgs]
        focal = all_cam['camera_params'][0][0] / 4.
        focal = torch.tensor((focal, focal), dtype=torch.float32)
        focal[0] *= tar_w / W
        focal[1] *= tar_h / H
        all_bboxes = None
        all_imgs = torch.stack(all_imgs)
        # all_poses = torch.Tensor(np.array(all_poses).astype(np.float32))
        all_poses = torch.stack(all_poses)
        c = torch.tensor((480, 320), dtype=torch.float32)
        # ========
        # all_imgs = F.interpolate(all_imgs, None, scale_factor=0.5, align_corners=True, mode='bilinear', recompute_scale_factor=False)
        # c *= 0.5
        # focal *= 0.5
        # =======
        scale = 0.1
        all_poses[:, :3, 3] *=scale
        all_masks = None
        bds = (np.array(all_cam['depth_ranges']).min()*scale, np.array(all_cam['depth_ranges']).max()*scale)
        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            "bds": bds
        }
        if all_masks is not None:
            result["masks"] = all_masks
        result["c"] = c
        return result
