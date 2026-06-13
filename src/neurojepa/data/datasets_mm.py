import sys
import logging
import warnings
import random
import numpy as np
import pandas as pd

from monai import data
from monai.utils.type_conversion import convert_data_type

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from neurojepa.data.transforms import loading_transforms
from neurojepa.masks.masking import MaskCollator

from typing import List, Dict, Any, Optional, Union, Tuple

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def custom_collate_fn(batch: List[Any]) -> Any:
    """
    Custom collate function to filter out None items from the batch
    and convert MONAI MetaTensors to regular PyTorch tensors.
    """
    batch = [item for item in batch if item is not None]
    
    # Convert MetaTensors to regular tensors to avoid resize issues
    def convert_metatensor(obj):
        if hasattr(obj, 'as_tensor'):
            # MONAI MetaTensor -> regular torch.Tensor
            return obj.as_tensor()
        elif isinstance(obj, dict):
            return {k: convert_metatensor(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            converted = [convert_metatensor(item) for item in obj]
            return type(obj)(converted)
        else:
            return obj
    
    batch = [convert_metatensor(item) for item in batch]
    return torch.utils.data.dataloader.default_collate(batch)


class BaseMRIDataset(Dataset):
    """
    Base dataset that maintains SEPARATE PersistentDataset caches for each modality.
    This allows reusing existing unimodal caches.
    
    Each modality gets its own cache, and __getitem__ retrieves from multiple caches.
    """

    def __init__(
        self,
        img_size: List[int],
        model_name: str,
        in_chans: int,
        csv_file: str,
        data_augmentation: Any,
        dataframe: Optional[pd.DataFrame] = None,
        cache_dir: Optional[Union[str, Dict[str, str]]] = None,
        all_cols: Optional[List[str]] = None,
        img_cols: Union[str, List[str]] = "img_path",
        img_flags: Optional[Union[str, List[str]]] = None,
        dropna: bool = False,
        drop_prob: float = 0.2,
    ):
        self.img_size = img_size
        self.drop_prob = drop_prob
        self.model_name = model_name
        self.in_chans = in_chans
        self.dropna = dropna

        # Convert img_cols to list if single string
        if isinstance(img_cols, str):
            self.img_cols = [img_cols]
        else:
            self.img_cols = list(img_cols)
            
        # Convert all_cols to list if single string
        if isinstance(all_cols, str):
            self.all_cols = [all_cols]
        else:
            self.all_cols = list(all_cols)
        
        # Convert img_flags to list if single string or None
        if img_flags is None:
            self.img_flags = [None] * len(self.img_cols)
        elif isinstance(img_flags, str):
            self.img_flags = [img_flags]
        else:
            self.img_flags = list(img_flags)
            
        if len(self.img_flags) != len(self.img_cols):
            raise ValueError(f"img_flags length must match img_cols length")

        self.num_modalities = len(self.img_cols)

        if not any(tag in self.model_name for tag in ("brainIAC", "jepa", "vit", "voco", "cnn")):
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # Load dataframe
        if dataframe is None:
            df = pd.read_csv(csv_file)
        else:
            df = dataframe
        
        # Check all image columns exist
        for col in self.img_cols:
            if col not in df.columns:
                raise ValueError(f"CSV must contain column '{col}'.")
        
        # Apply QC filtering for each modality
        for img_col, img_flag in zip(self.img_cols, self.img_flags):
            if img_flag is not None and img_flag in df.columns:
                df = df[df[img_flag] == True].reset_index(drop=True)
                print(f"Dropped rows with bad QC in '{img_flag}' for '{img_col}'. New dataset size: {len(df)}")
        
        # Set non-image columns (in all_cols but not in img_cols) to NaN
        if self.all_cols is not None:
            non_img_cols = [col for col in self.all_cols if col in df.columns and col not in self.img_cols]
            if non_img_cols:
                df[non_img_cols] = np.nan
                logger.info(f"Set columns {non_img_cols} to NaN.")
        
        cache_augmentation = loading_transforms(roi=self.img_size)
        self.data_augmentation = data_augmentation
        self.cache_dir = cache_dir
        self._paths_dict = {}
        for col in self.img_cols:
            # Handle NaN/None in dataframe for missing modalities
            if col in df.columns:
                # Convert to string, but keep NaN/None as None or empty string to identify them later if needed.
                self._paths_dict[col] = df[col].astype(str).tolist()
            else:
                # Should have been caught earlier, but just in case
                self._paths_dict[col] = ["nan"] * len(df)

        self._df = df

        # Create SEPARATE PersistentDataset for each modality
        self._cache_ds_dict = {}
        self._col_indices_map = {} # Maps global_idx -> sub_dataset_idx for each col (or -1 if missing)

        for col in self.img_cols:
            all_paths = self._paths_dict[col]
            
            # Identify which paths are valid (not 'nan', 'none', etc.)
            valid_data_list = []
            
            # Mapping from global index (dataframes's index) to PersistentDataset's local index
            # Initialize with -1 (missing)
            mapping = np.full(len(all_paths), -1, dtype=np.int32)
            
            current_sub_idx = 0
            for i, path in enumerate(all_paths):
                # Robust check for missing/nan paths
                if str(path).lower() in ['nan', 'none', 'null', '']:
                    continue
                
                # If valid
                valid_data_list.append({"image": path})
                mapping[i] = current_sub_idx
                current_sub_idx += 1
            
            self._col_indices_map[col] = mapping
            
            if len(valid_data_list) > 0:
                self._cache_ds_dict[col] = data.PersistentDataset(
                    data=valid_data_list,
                    transform=cache_augmentation,
                    cache_dir=self.cache_dir,
                )
            else:
                # Handle edge case where a column is entirely empty
                warnings.warn(f"Modality '{col}' has no valid samples! Is this expected?")
                self._cache_ds_dict[col] = None


        # Placeholder for any bad/failed sample
        self.placeholder_image = self._build_placeholder(self.img_size, self.in_chans)

    @staticmethod
    def _build_placeholder(img_size: Tuple[int, ...], in_chans: int) -> torch.Tensor:
        """Create a zero tensor placeholder as MONAI MetaTensor."""
        ph = torch.zeros((in_chans, *img_size), dtype=torch.float32)
        meta_tensor, *_ = convert_data_type(ph, output_type=data.MetaTensor)
        return meta_tensor

    def _valid_sample(self, sample: Dict[str, Any], idx: int, modality: str) -> bool:
        """Validate a single modality sample."""
        validity = True

        if "image" not in sample:
            warnings.warn(f"[idx={idx}, modality={modality}] Missing 'image' key.")
            validity = False
        else:
            img = sample["image"]
            if not isinstance(img, torch.Tensor):
                warnings.warn(f"[idx={idx}, modality={modality}] Image is not a tensor (got {type(img)}).")
                validity = False
            else:
                if img.ndim != (1 + len(self.img_size)):
                    warnings.warn(f"[idx={idx}, modality={modality}] Unexpected dims {tuple(img.shape)}.")
        return validity
    
    def __len__(self):
        return len(self._df)

    def _get_multimodal_sample(self, idx: int, data_augmentation: Any = None) -> Dict[str, Any]:
        multimodal_sample = {}
        # Identify valid modalities for this sample
        valid_cols = []
        possible_samples = {}
        
        for col in self.img_cols:
            # Use the index map to check validity and get local index for PersistentDataset
            sub_idx = self._col_indices_map[col][idx]
            
            if sub_idx == -1:
                # Path marked as missing/nan during init
                continue
                
            try:
                # cache_ds might be None if column was empty
                if self._cache_ds_dict[col] is None:
                    continue
                    
                # Get sample from this modality's cache using the local sub_idx
                sample = self._cache_ds_dict[col][sub_idx]
                
                if self._valid_sample(sample, idx, col):
                    possible_samples[col] = sample
                    valid_cols.append(col)
            except Exception as e:
                # If loading fails (e.g. file corrupted), treat as missing
                # warnings.warn(f"[idx={idx}, modality={col}] Error loading: {e}")
                pass

        # Apply Random Drop Augmentation (only if we have > 1 valid modality)
        if len(valid_cols) > 1 and random.random() < self.drop_prob:
            max_keep = len(valid_cols) - 1
            num_to_keep = random.randint(1, max_keep)
            
            # Choose valid_cols subset WITHOUT shuffling the list to preserve modality order
            keep_subset = set(random.sample(valid_cols, num_to_keep))
            valid_cols = [col for col in valid_cols if col in keep_subset]
            
        for col in self.img_cols:
            if col in valid_cols:
                # Retrieve and augment
                s = possible_samples[col]
                if data_augmentation is not None:
                    multimodal_sample[f"image_{col}"] = data_augmentation(s)["image"]
                else:
                    multimodal_sample[f"image_{col}"] = s["image"]
            else:
                # Missing or Dropped -> Placeholder
                multimodal_sample[f"image_{col}"] = self._fallback_image()
        
        # Add validity mask (1 for present, 0 for missing/dropped)
        multimodal_sample["__validity_mask__"] = torch.tensor(
            [1.0 if col in valid_cols else 0.0 for col in self.img_cols], 
            dtype=torch.float32
        )
        
        return multimodal_sample

    def _fallback_image(self) -> torch.Tensor:
        return self.placeholder_image
    
    
class FinetuneDataset(BaseMRIDataset):
    """
    Fine-tuning dataset that uses separate caches for each modality.
    Can reuse existing unimodal caches.
    """

    def __init__(
        self,
        img_size: List[int],
        model_name: str,
        in_chans: int,
        csv_file: str,
        data_augmentation: Any,
        label_col: str,
        num_classes: int,
        dataframe: Optional[pd.DataFrame] = None,
        cache_dir: Optional[Union[str, Dict[str, str]]] = None,
        all_cols: Optional[List[str]] = None,
        img_cols: Union[str, List[str]] = "img_path",
        img_flags: Optional[Union[str, List[str]]] = None,
        label_dtype: torch.dtype = torch.float32,
        na_as_zero: bool = True,
        dropna: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        super().__init__(
            img_size=img_size,
            model_name=model_name,
            in_chans=in_chans,
            csv_file=csv_file,
            dataframe=dataframe,
            data_augmentation=data_augmentation,
            cache_dir=cache_dir,
            all_cols=all_cols,
            img_cols=img_cols,
            img_flags=img_flags,
            dropna=dropna,
        )
        
        if self.dropna:
            cols_to_check = self.img_cols + [label_col]
            self._df = self._df.dropna(subset=cols_to_check).reset_index(drop=True)
            print(f"Dropped rows with NaN in {cols_to_check}. New dataset size: {len(self._df)}")
            self._paths_dict = {col: self._df[col].astype(str).tolist() for col in self.img_cols}
        
        if label_col not in self._df.columns:
            raise ValueError(f"Missing label column '{label_col}' in CSV.")

        labels_series = self._df[label_col].copy()
        if na_as_zero:
            labels_series = labels_series.fillna(0)

        # Handle regression vs multi-class vs binary labels
        if num_classes == 1:
            labels_series = labels_series.astype(float)
            if mean is not None and std is not None:
                labels_series = (labels_series - mean) / std
            self.label_arr = labels_series
            self.label_dtype = torch.float32
        elif num_classes == 2:
            self.label_arr = (labels_series.astype(float) > 0.5).astype(np.int64)
            self.label_dtype = torch.float32
        elif num_classes > 2:
            self.label_arr = labels_series.astype(np.int64)
            self.label_dtype = torch.long
        else:
            raise ValueError("num_classes must be >= 1")

        self.label_col = str(label_col)
        self.num_classes = int(num_classes)

        if self.num_classes > 1:
            class_counts = self.label_arr.value_counts()
            for i in range(num_classes):
                if i not in class_counts or class_counts[i] == 0:
                    warnings.warn(f"Class {i} has zero samples in the dataset.")
                    

    def __getitem__(self, idx: int):
        try:
            # Get samples from all modality caches
            multimodal_sample = self._get_multimodal_sample(idx, data_augmentation=self.data_augmentation)

            # Build target
            target_val = self.label_arr.iloc[idx]
            target = torch.tensor(target_val, dtype=self.label_dtype)
            
            return multimodal_sample, target
            
        except Exception as e:
            warnings.warn(f"[idx={idx}] Error during load/augment: {e}")
            return {f"image_{col}": self.placeholder_image for col in self.img_cols}, torch.tensor(0, dtype=self.label_dtype)

        
        
def get_finetune_dataset(cfg: Any, augs: Any):
    """
    Get dataloaders for fine-tuning with class balancing for the training set.
    Uses FinetuneDataset for single-label, multi-class classification.
    """
    # Unpack augmentations
    imtrans, imvals, imtests = augs[0], augs[1], augs[2]
    
    # Get data parameters
    batch_size = cfg.data.batch_size
    cache_dir = cfg.data.cache_dir if cfg.data.cache_dir else None

    # Get distributed parameters
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    
    # Class Balancing for Training Set 
    df_train = pd.read_csv(cfg.data.train_csv_path)
    df_val = pd.read_csv(cfg.data.val_csv_path)
    df_test = pd.read_csv(cfg.data.test_csv_path)
    
    # get None or value from data
    all_cols = cfg.data.all_cols
    img_cols = cfg.data.img_cols
    label_col = cfg.data.label_col
    num_classes = cfg.data.num_classes
    
    # Clean dataset
    df_train = clean_dataset(df_train, img_cols, label_col)
    df_val = clean_dataset(df_val, img_cols, label_col)
    df_test = clean_dataset(df_test, img_cols, label_col)

    # Get label array (single column of class indices)
    labels_series = df_train[label_col].copy().fillna(0)
    label_arr = labels_series.astype(np.int64)

    # Calculate class weights (inverse frequency)
    if num_classes > 1:
        eps = 1e-6
        class_counts = np.bincount(label_arr, minlength=num_classes)
        N = len(label_arr)
        class_weights = N / (class_counts + eps)

        # Assign weight to each sample based on its class
        sample_weights = class_weights[label_arr]
        
        # Normalize weights
        sample_weights = sample_weights / (sample_weights.mean() + eps)
    else:
        sample_weights = np.ones(len(label_arr), dtype=np.float32)
        class_counts = None
        class_weights = None
        
    logger.info(f"Finetune Class counts: {class_counts}")
    logger.info(f"Finetune Class weights: {class_weights}")
    
    # Get model parameters
    img_size = cfg.model.img_size
    model_name = cfg.model.model_name
    in_chans = cfg.model.in_chans
    # Get sample size
    sample_size = cfg.data.samples_per_epoch
    sample_size = None if sample_size == -1 else sample_size // num_tasks
    # Train Dataloader
    # calculate mean and std for regression
    if num_classes == 1:
        mean = labels_series.mean()
        std = labels_series.std()
        logger.info(f"Regression label mean: {mean}, std: {std}")
    else:
        mean, std = None, None
    
    train_ds = FinetuneDataset(
        img_size=list(img_size),
        model_name=str(model_name),
        in_chans=int(in_chans),
        csv_file=cfg.data.train_csv_path, 
        dataframe=df_train,
        data_augmentation=imtrans, 
        all_cols=all_cols,
        img_cols=img_cols,
        label_col=label_col,
        num_classes=num_classes,
        cache_dir=cache_dir,
        dropna=False,
        # mean=mean,
        # std=std,
    )
    
    sampler_train = data.DistributedWeightedRandomSampler(
        dataset=train_ds,
        weights=sample_weights,
        num_samples_per_rank=sample_size, 
        rank=global_rank, 
    )
    train_loader = data.ThreadDataLoader(
        dataset=train_ds,
        batch_size=batch_size, 
        sampler=sampler_train,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )

    # Validation Dataloader
    val_ds = FinetuneDataset(
        img_size=list(img_size),
        model_name=str(model_name),
        in_chans=int(in_chans),
        csv_file=cfg.data.val_csv_path, 
        dataframe=df_val,
        data_augmentation=imvals, 
        all_cols=all_cols,
        img_cols=img_cols,
        label_col=label_col,
        num_classes=num_classes,
        cache_dir=cache_dir,
        dropna=True,
    )
    sampler_val = data.DistributedSampler(
        dataset=val_ds, shuffle=False, num_replicas=num_tasks, rank=global_rank
    )
    val_loader = data.ThreadDataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        sampler=sampler_val,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )

    # Test Dataloader
    test_ds = FinetuneDataset(
        img_size=list(img_size),
        model_name=str(model_name),
        in_chans=int(in_chans),
        csv_file=cfg.data.test_csv_path, 
        dataframe=df_test,
        data_augmentation=imtests, 
        all_cols=all_cols,
        img_cols=img_cols,
        label_col=label_col,
        num_classes=num_classes,
        cache_dir=cache_dir,
        dropna=True,
    )
    sampler_test = data.DistributedSampler(
        dataset=test_ds, shuffle=False, num_replicas=num_tasks, rank=global_rank
    )
    test_loader = data.ThreadDataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        sampler=sampler_test,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )
    
    return train_loader, val_loader, test_loader


class PretrainDataset(BaseMRIDataset):
    """
    Multimodal dataset for self-supervised pretraining (no labels).
    
    Returns a dict of image tensors, one per modality:
        { "image_{col_name}": torch.Tensor [C, *img_size], ... }
    """

    def __getitem__(self, idx: int):
        try:
            multimodal_sample = self._get_multimodal_sample(idx, data_augmentation=self.data_augmentation)
            return multimodal_sample
        except Exception as e:
            warnings.warn(f"[idx={idx}] Error during load/augment: {e}")
            return {f"image_{col}": self.placeholder_image for col in self.img_cols}


def multimodal_mask_collate_fn(batch: List[Dict[str, torch.Tensor]], mask_collator: MaskCollator) -> Any:
    """
    Collate function for multimodal pretraining that:
      1. Filters out None items and converts MetaTensors
      2. Stacks each modality separately
      3. Generates masks (shared across modalities)
    
    Returns:
        Tuple of (images_dict, masks_enc, masks_pred) where:
            images_dict: dict of {modality_key: Tensor [B, C, *img_size]}
            masks_enc: list of encoder masks from MaskCollator
            masks_pred: list of predictor masks from MaskCollator
    """
    # Filter Nones and convert MetaTensors
    batch = [item for item in batch if item is not None]
    
    def convert_metatensor(obj):
        if hasattr(obj, 'as_tensor'):
            return obj.as_tensor()
        elif isinstance(obj, dict):
            return {k: convert_metatensor(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)([convert_metatensor(item) for item in obj])
        return obj
    
    batch = [convert_metatensor(item) for item in batch]
    
    if len(batch) == 0:
        return None
    
    # Stack each modality: {key: [B, C, *img_size]}
    modality_keys = list(batch[0].keys())
    
    # Separate special keys
    special_keys = ["__validity_mask__"]
    image_keys = [k for k in modality_keys if k not in special_keys]
    
    stacked_images = {}
    for key in image_keys:
        stacked_images[key] = torch.stack([sample[key] for sample in batch])
        
    # Stack validity mask if present
    validity_mask = None
    if "__validity_mask__" in batch[0]:
        validity_mask = torch.stack([sample["__validity_mask__"] for sample in batch])
    
    # Generate masks using the MaskCollator (only needs batch_size, spatial dims are fixed)
    batch_size = len(batch)
    masks_enc, masks_pred = [], []
    for mask_generator in mask_collator.mask_generators:
        m_enc, m_pred = mask_generator(batch_size)
        masks_enc.append(m_enc)
        masks_pred.append(m_pred)
    
    return (stacked_images, masks_enc, masks_pred, validity_mask)


def get_pretrain_dataloaders(cfg: Any, augs: Any):
    """
    Get dataloader for multimodal self-supervised pretraining.
    
    Args:
        cfg: configuration object with data, model, mask sections.
        augs: augmentation transforms for pretraining.
    
    Returns:
        (train_loader, mask_collator)
    """
    # Dataset parameters
    batch_size = cfg.data.batch_size
    cache_dir = cfg.data.cache_dir if cfg.data.cache_dir else None
    csv_file = cfg.data.train_csv_path

    # Distributed parameters
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    # Model and data parameters
    img_size = cfg.model.img_size
    patch_size = cfg.model.patch_size
    model_name = cfg.model.model_name
    in_chans = cfg.model.in_chans
    all_cols = cfg.data.all_cols
    img_cols = cfg.data.img_cols

    # Read and clean CSV (drop rows where ALL image columns are NaN)
    df = pd.read_csv(csv_file)
    
    # Standardize missing values: convert 'None', 'nan', 'null' strings to np.nan
    for col in img_cols:
        if col in df.columns:
             df[col] = df[col].replace(r'(?i)^(none|nan|null|\s*)$', np.nan, regex=True)

    # Filter out rows that have no valid image paths at all
    df = df.dropna(subset=img_cols, how='all').reset_index(drop=True)
    
    # Apply QC filtering
    img_flags = cfg.data.get("img_flags", None)
    if img_flags is not None:
        if isinstance(img_flags, str):
            img_flags = [img_flags]
        for flag_col in img_flags:
            if flag_col in df.columns:
                df = df[df[flag_col] == True].reset_index(drop=True)
    
    logger.info(f"Multimodal pretrain dataset: {len(df)} samples, {len(img_cols)} modalities: {img_cols}")

    mask_collator = MaskCollator(
        cfgs_mask=cfg.mask,
        crop_size=img_size,
        patch_size=patch_size,
    )

    train_ds = PretrainDataset(
        img_size=list(img_size),
        model_name=str(model_name),
        in_chans=int(in_chans),
        csv_file=csv_file,
        dataframe=df,
        data_augmentation=augs,
        cache_dir=cache_dir,
        all_cols=all_cols,
        img_cols=img_cols,
    )

    sampler_train = data.DistributedSampler(
        train_ds,
        shuffle=True,
        num_replicas=num_tasks,
        rank=global_rank,
    )

    # Wrap mask_collator for multimodal batches
    def collate_fn(batch):
        return multimodal_mask_collate_fn(batch, mask_collator)

    train_loader = data.ThreadDataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        sampler=sampler_train,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
        drop_last=True,
    )

    return train_loader, mask_collator


# Quality control clean modality utility
def clean_dataset(df, img_cols, label_col):
    for img_col in img_cols:
        if img_col not in df.columns:
            raise ValueError(f"CSV must contain column '{img_col}'.")
        else:
            df = df.dropna(subset=[img_col, label_col]).reset_index(drop=True)
            
            # filter based on qc columns if they exist
            if 't1w' in img_col and 't1w_qc' in df.columns:
                df = df.dropna(subset=['t1w_qc'])
                df = df[df['t1w_qc'] == False]
            elif 't2w' in img_col and 't2w_qc' in df.columns:
                df = df.dropna(subset=['t2w_qc'])
                df = df[df['t2w_qc'] == False]
            elif 'flair' in img_col and 'flair_qc' in df.columns:
                df = df.dropna(subset=['flair_qc'])
                df = df[df['flair_qc'] == False]

            print(f"Clean dataset. New dataset size: {len(df)}")
                
    return df.reset_index(drop=True)