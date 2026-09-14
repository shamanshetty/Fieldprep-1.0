"""
Central configuration for potato disease classification pipeline.
"""
import os
import random
import cv2


# ============ Global / shared settings (used by train/evaluate/predict/xai) ============
IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PreprocessingConfig:
    # --- Paths ---
    # Set this to the directory containing one subdirectory per disease class.
    INPUT_DIR = "raw_dataset"
    OUTPUT_DIR = "processed_dataset"
    DUPLICATES_DIR = "duplicates"

    # --- Duplicate handling ---
    MOVE_DUPLICATES = True  # move to DUPLICATES_DIR instead of deleting

    # --- Resize ---
    TARGET_SIZE = IMAGE_SIZE
    INTERPOLATION_METHOD = cv2.INTER_LANCZOS4

    # --- Normalization (applied during preprocessing, separate from train-time MEAN/STD) ---
    APPLY_NORMALIZATION = False
    NORMALIZATION_METHOD = 'standardize'  # 'minmax' or 'standardize'
    NORMALIZATION_MEAN = MEAN
    NORMALIZATION_STD = STD

    # --- Train/val/test split ---
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # --- Reproducibility / misc ---
    RANDOM_SEED = 42
    VERBOSE = True
    TEST_MODE = False
    TEST_SIZE = 20  # used only when TEST_MODE is True

    # --- Augmentation target ---
    TARGET_IMAGES_PER_CLASS = 1500
    MAX_AUG_PER_IMAGE = 10

    # --- Geometric augmentation toggles/ranges ---
    APPLY_ROTATION = True
    ROTATION_ANGLES = [-25, -15, -10, 10, 15, 25]

    APPLY_FLIP = True
    FLIP_HORIZONTAL = True
    FLIP_VERTICAL = False
    FLIP_BOTH = False

    APPLY_TRANSLATION = True
    TRANSLATION_RANGE = 0.1  # fraction of image dimension

    APPLY_SCALE = True
    SCALE_RANGE = (0.85, 1.15)

    APPLY_SHEAR = True
    SHEAR_RANGE = 12  # degrees

    # --- Photometric / noise augmentation ---
    COLOR_JITTER_BRIGHTNESS = 0.2
    COLOR_JITTER_CONTRAST = 0.2
    COLOR_JITTER_SATURATION = 0.2

    GAUSSIAN_NOISE_STD = 10.0
    BLUR_KERNEL_SIZES = [3, 5]
    RANDOM_CROP_RANGE = (0.8, 0.95)

    # --- Augmentation op weights (used to build a per-class plan) ---
    AUGMENTATION_OPS_WEIGHTS = {
        'rotation': 0.20,
        'flip': 0.15,
        'translation': 0.10,
        'scale': 0.10,
        'shear': 0.10,
        'color_jitter': 0.15,
        'gaussian_noise': 0.10,
        'blur': 0.05,
        'random_crop': 0.05,
    }

    @classmethod
    def validate(cls):
        """Sanity-check configuration values before running the pipeline."""
        if not os.path.isdir(cls.INPUT_DIR):
            raise FileNotFoundError(f"INPUT_DIR does not exist: {cls.INPUT_DIR}")

        ratio_sum = cls.TRAIN_RATIO + cls.VAL_RATIO + cls.TEST_RATIO
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0, got {ratio_sum}"
            )

        if cls.TARGET_IMAGES_PER_CLASS <= 0:
            raise ValueError("TARGET_IMAGES_PER_CLASS must be positive")

        return True

    @classmethod
    def print_config(cls):
        print("Preprocessing configuration:")
        print(f"  INPUT_DIR: {cls.INPUT_DIR}")
        print(f"  OUTPUT_DIR: {cls.OUTPUT_DIR}")
        print(f"  DUPLICATES_DIR: {cls.DUPLICATES_DIR}")
        print(f"  TARGET_SIZE: {cls.TARGET_SIZE}")
        print(f"  TRAIN/VAL/TEST RATIO: {cls.TRAIN_RATIO}/{cls.VAL_RATIO}/{cls.TEST_RATIO}")
        print(f"  TARGET_IMAGES_PER_CLASS: {cls.TARGET_IMAGES_PER_CLASS}")
        print(f"  RANDOM_SEED: {cls.RANDOM_SEED}")

    @classmethod
    def compute_augmentation_plan(cls, current_count):
        """
        Build a dict {operation_name: count} describing how many augmented
        images to generate via each operation so the class reaches
        TARGET_IMAGES_PER_CLASS, weighted by AUGMENTATION_OPS_WEIGHTS.
        """
        needed = max(0, cls.TARGET_IMAGES_PER_CLASS - current_count)
        if needed == 0:
            return {}

        # Only include ops that are enabled via their APPLY_* toggle
        enabled_ops = {
            'rotation': cls.APPLY_ROTATION,
            'flip': cls.APPLY_FLIP,
            'translation': cls.APPLY_TRANSLATION,
            'scale': cls.APPLY_SCALE,
            'shear': cls.APPLY_SHEAR,
            'color_jitter': True,
            'gaussian_noise': True,
            'blur': True,
            'random_crop': True,
        }

        active_weights = {
            op: w for op, w in cls.AUGMENTATION_OPS_WEIGHTS.items() if enabled_ops.get(op, False)
        }
        weight_total = sum(active_weights.values())
        if weight_total == 0:
            return {'rotation': needed}

        plan = {}
        allocated = 0
        ops = list(active_weights.items())
        for i, (op, weight) in enumerate(ops):
            if i == len(ops) - 1:
                count = needed - allocated
            else:
                count = int(round(needed * (weight / weight_total)))
                allocated += count
            if count > 0:
                plan[op] = count

        return plan

    @classmethod
    def select_images_to_keep(cls, files, target):
        """
        Deterministically select `target` files to keep out of `files`
        (used when downsampling an oversized class).
        """
        rng = random.Random(cls.RANDOM_SEED)
        files_sorted = sorted(files)
        rng.shuffle(files_sorted)
        return files_sorted[:target]


class TrainingConfig:
    # Supported torchvision backbones
    AVAILABLE_MODELS = [
        'resnet34', 'resnet50', 'mobilenet_v2',
    ]
    MODEL_NAME = 'mobilenet_v2'

    EPOCHS = 20
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32

    # Optional custom pretrained base weights / partial freezing
    BASE_MODEL_PATH = None
    FREEZE_LAYERS_UNTIL = None

    # LR scheduler
    LR_SCHEDULER_STEP_SIZE = 10
    LR_SCHEDULER_GAMMA = 0.1

    # Early stopping
    APPLY_EARLY_STOPPING = True
    EARLY_STOPPING_MIN_DELTA = 0.001
    EARLY_STOPPING_PATIENCE = 5
