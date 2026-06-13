import numpy as np
from typing import Tuple,  Any

from monai import transforms
from monai.transforms import Lambdad


def remove_nan(img):
    img[np.isnan(img)]=0.0
    return img

def loading_transforms(
        roi: Tuple[int, int, int], 
        spacing: Tuple[float, float, float]=(1.0, 1.0, 1.0),
        model_name: str = 'vit',
    ) -> transforms.Compose:
    """
    Define loading transforms based on the number of input channels.

    Args:
        roi (Tuple[int, int, int]): Region of interest size.
        spacing (Tuple[float, float, float]): Desired spacing for resampling.
        model_name (str): Name of the model to determine specific transforms.

    Returns:
        transforms.Compose: Composed transforms.
    """
    trans = transforms.Compose(
        [
            transforms.LoadImaged(
                keys=["image", "label"],
                image_only=False,
                allow_missing_keys=True,
            ),
            transforms.EnsureChannelFirstd(
                keys=["image", "label"],
                allow_missing_keys=True,
            ),
            # remove nan
            Lambdad(("image",), remove_nan), 
            transforms.Orientationd(
                keys=["image", "label"],
                axcodes="RAS",
                labels=(('L', 'R'), ('P', 'A'), ('I', 'S')),
                allow_missing_keys=True,
            ),
            transforms.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
                allow_missing_keys=True,
            ),
            transforms.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=[180, 216, 180],
                mode="edge",
                allow_missing_keys=True,
            ),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=[5, 'nearest'],
                allow_missing_keys=True,
            ),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                select_fn=lambda x: x > 0.0,
                margin=4,
                allow_smaller=True,
                allow_missing_keys=True,
            ),
            transforms.Resized(
                keys=["image", "label"],
                spatial_size=roi,
                allow_missing_keys=True,
            ),
        ]
    )
    return trans


def jepa3d_transforms(config: Any, mode: str = 'train', reshape: bool = False) -> transforms.Compose:
    """
    Define MAE3D transforms based on the mode and reshape flag.

    Args:
        config (Any): Configuration object.
        mode (str): Mode of operation ('train', 'val', 'test').
        reshape (bool): Whether to reshape the image.

    Returns:
        transforms.Compose: Composed transforms.
    """
    roi_size = config.data.img_size
    if mode in ['train', 'val']:
        trans = transforms.Compose([
            transforms.CastToTyped(
                keys=["image"],
                dtype=np.float32,
                allow_missing_keys=True,
            ),
            transforms.CenterSpatialCropd(
                keys=["image"],
                roi_size=roi_size,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image"],
                prob=0.5,
                spatial_axis=0,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image"],
                prob=0.5,
                spatial_axis=1,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image"],
                prob=0.5,
                spatial_axis=2,
                allow_missing_keys=True,
            ),
            transforms.RandBiasFieldd(
                keys=["image"], 
                prob=0.5, 
                coeff_range=(0.0, 0.1), 
                degree=3,
                allow_missing_keys=True,
            ),
            transforms.RandAdjustContrastd(
                keys=["image"], 
                gamma=(0.6, 1.5), 
                prob=0.5, 
                allow_missing_keys=True,
            ),
            transforms.RandGaussianNoised(
                keys=["image"], 
                mean=0.0,
                std=0.1,
                prob=0.5, 
                allow_missing_keys=True,
            ),
            transforms.RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=0.5,
                allow_missing_keys=True,
            ),
            transforms.ToTensord(
                keys=["image"],
                allow_missing_keys=True,
            ),
        ])
    elif mode == 'test':
        trans = transforms.Compose([
            transforms.CenterSpatialCropd(
                keys=["image"],
                roi_size=roi_size,
                allow_missing_keys=True,
            ),
            transforms.ToTensord(
                keys=["image"],
                allow_missing_keys=True,
            ),
        ])
    else:
        raise NotImplementedError(f"{mode} mode not implemented.")

    return trans


def vit3d_transforms(config: Any, mode: str = 'train') -> transforms.Compose:
    """
    Define ViT transforms based on the mode.

    Args:
        config (Any): Configuration object.
        mode (str): Mode of operation ('train', 'val', 'test').

    Returns:
        transforms.Compose: Composed transforms.
    """
    roi_size = config.data.img_size
    if mode == 'train':
        trans = transforms.Compose([
            transforms.CastToTyped(
                keys=["image", "label"],
                dtype=np.float32,
                allow_missing_keys=True,
            ),
            # pad
            transforms.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=roi_size,
                mode="constant",        # padding mode
                value=0,                # padding value
                allow_missing_keys=True,
            ),
            # central crop
            transforms.CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=roi_size,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image", "label"],
                prob=0.1,
                spatial_axis=0,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image", "label"],
                prob=0.1,
                spatial_axis=1,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image", "label"],
                prob=0.1,
                spatial_axis=2,
                allow_missing_keys=True,
            ),
            transforms.RandAdjustContrastd(
                keys=["image"], 
                gamma=(0.6, 1.5), 
                prob=0.5, 
                allow_missing_keys=True,
            ),
            transforms.RandGaussianNoised(
                keys=["image"], 
                mean=0.0,
                std=0.1,
                prob=0.5, 
                allow_missing_keys=True,
            ),
            transforms.RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=0.5,
                allow_missing_keys=True,
            ),
            transforms.ToTensord(
                keys=["image", "label"],
                allow_missing_keys=True,
            ),
        ])
    elif mode in ['val', 'test']:
        trans = transforms.Compose([
            transforms.CastToTyped(
                keys=["image", "label"],
                dtype=np.float32,
                allow_missing_keys=True,
            ),
            transforms.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=roi_size,
                mode="constant",        # padding mode
                value=0,                # padding value
                allow_missing_keys=True,
            ),
            # central crop
            transforms.CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=roi_size,
                allow_missing_keys=True,
            ),
            transforms.ToTensord(
                keys=["image", "label"],
                allow_missing_keys=True,
            ),
        ])
    else:
        raise NotImplementedError(f"{mode} mode not implemented.")

    return trans
