"""Resolution-Aware Augmentation Strategy for Aerial Imagery.
Key innovation from Solafune competition experiments:
- High-resolution tiles (10-20 cm): Zoom-out simulates distant altitude context.
- Mid/low-resolution tiles (30-50 cm): Disables zoom-out to prevent loss of sub-pixel crown detail.
"""

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def get_resolution_aware_transforms(resolution_cm: float, is_training: bool = True):
    """Returns resolution-specific Albumentations pipeline.
    
    Args:
        resolution_cm: Ground sampling distance (GSD) in cm (e.g. 10.0 to 80.0)
        is_training: Whether transforms include data augmentation or validation normalization
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError(
            "albumentations is required for transforms. Install with: pip install albumentations"
        )

    if not is_training:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5)
    ]

    # Apply zoom-out context simulation ONLY to high-resolution images
    if resolution_cm <= 25.0:
        transforms.append(
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3)
        )
    else:
        # For lower resolution images, preserve crown contrast and edge sharpness
        transforms.append(
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3)
        )

    transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return A.Compose(transforms)
