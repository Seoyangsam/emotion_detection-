
import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 4
# START_TRAIN_AT_IMG_SIZE = 512
DATASET = '../datasets_128_128'
# DATASET = 'datasets_512_512'
# CHECKPOINT_GEN = "generator.pth"
# CHECKPOINT_CRITIC = "critic.pth"
# CHECKPOINT_GEN = "saved_models/generator.pth"
# CHECKPOINT_CRITIC = "saved_models/critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
# LOAD_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3

BATCH_SIZES = [128, 128, 128, 64, 64, 64, 16, 8]
             #  4,  8,  16,  32, 64, 128,256,512
CHANNELS_IMG = 3
Z_DIM = 256  # should be 512 in original paper
IN_CHANNELS = 256  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10

PROGRESSIVE_EPOCHS = [30,30,60,120,120,240,120,120]
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4