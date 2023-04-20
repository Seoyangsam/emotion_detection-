import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config
import os

torch.backends.cudnn.benchmarks = True

gen_picture_size = 128
saved_generator_models_path = f'saved_models_{gen_picture_size}/generator_3nice.pth'

gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
load_checkpoint(saved_generator_models_path, gen, opt_gen, config.LEARNING_RATE,)
step = int(log2(gen_picture_size / 4))

if not os.path.exists(f'generate_picture_{gen_picture_size}'):
    os.makedirs(f'generate_picture_{gen_picture_size}')

save_path = f'generate_picture_{gen_picture_size}/img_'

generate_examples(gen, step, save_path=save_path, truncation=0.7, n=800)



