from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
from utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

class BubbleDataset(Dataset):
    def __init__(self, args, mask_root=None, image_root=None, func=None, transform=True, cropping=True):
        self.mask_root = mask_root
        self.image_root = image_root
        self.transform = transform
        self.split = [0.99, 0.01] # validation data is only used for output training samples
        self.seed = 213
        self.args = args
        self.func = func
        self.cropping = cropping
        np.random.seed(self.seed)

        self.masks = sorted(os.listdir(mask_root), key=lambda x: int(''.join(filter(str.isdigit, x))))
        self.images = sorted(os.listdir(image_root), key=lambda x: int(''.join(filter(str.isdigit, x))))

        self.mask_len = len(self.masks)
        self.images_len = len(self.images)


        train_images, val_images = self.images[:int(self.split[0] * self.images_len)], self.images[int(
            self.split[0] * self.images_len):]
        train_masks, val_masks = self.masks[:int(self.split[0] * self.mask_len)], self.masks[int(
            self.split[0] * self.mask_len):]

        if self.func == 'train':
            self.images = train_images
            self.masks = train_masks
        elif self.func == 'validation':
            self.images = val_images
            self.masks = val_masks
        elif self.func == 'testing':
            self.images = self.images
            self.masks = self.masks


        self.mask_len = len(self.masks)
        self.images_len = len(self.images)
        self.len_dataset = max(self.mask_len,self.images_len)

    # New for center crop
    def _get_fg_coords(self, mask: np.ndarray):
        """
        返回前景像素坐标 (ys, xs)。
        假设背景为 0（黑色）；RGB mask 任意通道非0都算前景。
        """
        if mask.ndim == 2:  # grayscale
            fg = mask > 0
        else:  # RGB
            fg = np.any(mask != 0, axis=-1)
        ys, xs = np.where(fg)
        return ys, xs
    # New for center crop
    def _crop_by_topleft(self, arr: np.ndarray, y1: int, x1: int, crop_size: int):
        return arr[y1:y1 + crop_size, x1:x1 + crop_size]
    # New for center crop
    def _center_to_topleft(self, h: int, w: int, cy: int, cx: int, crop_size: int):
        half = crop_size // 2
        y1 = int(cy - half)
        x1 = int(cx - half)
        y1 = max(0, min(y1, h - crop_size))
        x1 = max(0, min(x1, w - crop_size))
        return y1, x1
    # New for center crop
    def _fg_ratio(self, mask: np.ndarray):
        """计算前景比例（0~1）"""
        if mask.ndim == 2:
            fg = mask > 0
        else:
            fg = np.any(mask != 0, axis=-1)
        return float(fg.mean())
    # New for center crop
    def _best_crop_by_gradient(self, image: np.ndarray, crop_size: int, tries: int = 10):
        """
        用灰度梯度/纹理强度引导 crop：
        随机采样多个候选crop，选择梯度能量最大的那个。
        这样更可能裁到气泡边缘/纹理区域，而不是大片平坦背景。
        """
        h, w = image.shape[:2]
        if h < crop_size or w < crop_size:
            raise ValueError(f"Image too small for crop: {image.shape} vs crop_size={crop_size}")

        # 转灰度
        if image.ndim == 2:
            gray = image.astype(np.float32)
        else:
            # RGB -> gray
            gray = (0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]).astype(np.float32)

        # 计算简单梯度能量（不用cv2也行）
        gy = np.abs(gray[1:, :] - gray[:-1, :])
        gx = np.abs(gray[:, 1:] - gray[:, :-1])
        # pad到同尺寸
        gy = np.pad(gy, ((0, 1), (0, 0)), mode='edge')
        gx = np.pad(gx, ((0, 0), (0, 1)), mode='edge')
        grad = gx + gy  # 梯度幅值近似

        best_score = -1.0
        best_y1, best_x1 = 0, 0

        for _ in range(max(1, tries)):
            x1 = np.random.randint(0, w - crop_size + 1)
            y1 = np.random.randint(0, h - crop_size + 1)
            patch_grad = grad[y1:y1 + crop_size, x1:x1 + crop_size]
            score = float(patch_grad.mean())  # 也可以用 sum/percentile
            if score > best_score:
                best_score = score
                best_y1, best_x1 = y1, x1

        return best_y1, best_x1



    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        image_dir = os.path.join(self.image_root,self.images[idx % self.images_len])
        mask_dir = os.path.join(self.mask_root,self.masks[idx % self.mask_len])

        if self.args.image_channel == 1:
            image_type = 'L'
        elif self.args.image_channel == 3:
            image_type = 'RGB'

        if self.args.mask_channel == 1:
            mask_type = 'L'
        elif self.args.mask_channel == 3:
            mask_type = 'RGB'


        image = np.array(Image.open(image_dir).convert(image_type))
        mask = np.array(Image.open(mask_dir).convert(mask_type))

        # Resize if enabled
        if self.args.resize:
            resize_transform = A.Compose(
                [
                    A.Resize(height=self.args.resize_size, width=self.args.resize_size)
                ],
                additional_targets={'mask': 'mask'}
            )
            resized = resize_transform(image=image, mask=mask)
            image = resized['image']
            mask = resized['mask']

        # New part for center crop
        if self.cropping:

            # check image and mask after crop
            # image_before = image.copy()
            # mask_before  = mask.copy()

            crop_size = self.args.crop_size

            # ---------- 1) image：用梯度/纹理强度引导 crop（提升命中气泡区域概率） ----------
            y1_img, x1_img = self._best_crop_by_gradient(
                image=image,
                crop_size=crop_size,
                tries=getattr(self.args, "img_crop_tries", 10)  # 没有就默认10
            )
            image = self._crop_by_topleft(image, y1_img, x1_img, crop_size)

            # ---------- 2) mask：前景中心 crop + 前景比例阈值 + 重试 ----------
            h2, w2 = mask.shape[:2]
            if h2 < crop_size or w2 < crop_size:
                raise ValueError(f"Mask too small for crop: {mask.shape} vs crop_size={crop_size}")

            ys, xs = self._get_fg_coords(mask)

            # 这些阈值你可以放 args 里，不放也能跑
            min_fg_ratio = getattr(self.args, "min_fg_ratio", 0.30)   # 例如 30%
            max_tries = getattr(self.args, "mask_crop_tries", 8)      # 重试次数

            # fallback：随机起点
            def random_topleft():
                x1 = np.random.randint(0, w2 - crop_size + 1)
                y1 = np.random.randint(0, h2 - crop_size + 1)
                return y1, x1

            chosen_y1, chosen_x1 = random_topleft()

            if len(xs) > 0:
                # 尝试多次：每次随机选一个前景像素当中心，裁出来看前景比例够不够
                for _ in range(max_tries):
                    k = np.random.randint(0, len(xs))
                    cy, cx = int(ys[k]), int(xs[k])
                    y1, x1 = self._center_to_topleft(h2, w2, cy, cx, crop_size)
                    cand = self._crop_by_topleft(mask, y1, x1, crop_size)
                    if self._fg_ratio(cand) >= min_fg_ratio:
                        chosen_y1, chosen_x1 = y1, x1
                        break
                else:
                    # 如果多次都达不到阈值，就用“前景点中心裁剪”的最后一次，至少保证裁到前景附近
                    chosen_y1, chosen_x1 = y1, x1
            # 如果完全没有前景，退化随机裁剪
            mask = self._crop_by_topleft(mask, chosen_y1, chosen_x1, crop_size)

            # check image and mask after crop
            # if idx == 0:  # 只看一张
            #     plt.figure(figsize=(6,6))
            #     plt.subplot(221); plt.title('img before'); plt.imshow(image_before, cmap='gray'); plt.axis('off')
            #     plt.subplot(222); plt.title('img after');  plt.imshow(image, cmap='gray');       plt.axis('off')
            #     plt.subplot(223); plt.title('mask before');plt.imshow(mask_before);             plt.axis('off')
            #     plt.subplot(224); plt.title('mask after'); plt.imshow(mask);                    plt.axis('off')
            #     plt.show()

        if self.transform:
            img_trans = A.Compose([
                A.Normalize(mean=[0.5] * self.args.image_channel, std=[0.5] * self.args.image_channel,
                            max_pixel_value=255),
                ToTensorV2(),
            ])

            mask_trans = A.Compose([
                A.Normalize(mean=[0.5] * self.args.mask_channel, std=[0.5] * self.args.mask_channel,
                            max_pixel_value=255),
                ToTensorV2(),
            ])
            transform = img_trans(image=image)
            image = transform['image']
            transform = mask_trans(image=mask)
            mask = transform['image']

        return image, mask
    
    
