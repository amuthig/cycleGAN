import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data"
VAL_DIR = "data"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10.0
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = False
CHECKPOINT_GEN_A = "saved_model/genA.pth.tar"
CHECKPOINT_GEN_B = "saved_model/genB.pth.tar"
CHECKPOINT_CRITIC_A = "saved_model/criticA.pth.tar"
CHECKPOINT_CRITIC_B = "saved_model/criticB.pth.tar"

transforms = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
