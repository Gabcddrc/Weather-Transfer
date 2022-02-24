from comet_ml import Experiment
import torch.nn as nn
import torch.nn.functional as F
from model import *
from torch.autograd import Variable
import itertools

import torch 
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

import pickle
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))


BATCH_SIZE = 1
EPOCHS = 200
BETA = (0.5, 0.999)
LR = 2e-4
DECAY = 100
NUM_RESBLOCK  = 9

experiment = Experiment(
    api_key="HyWl3zINDt5tVxrEffWKpJ0Ms",
    project_name="deep-learning",
    workspace="jjiang",
)
experiment.set_name('CycleGAN-9-resblock')
# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": LR,
    "model" : 'CycleGAN',
    "EPOCHS": EPOCHS,
    "batch_size": BATCH_SIZE,
}
experiment.log_parameters(hyper_params)

def load_weather(name):
    with open('/cluster/scratch/zhejiang/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

overcast = load_weather('overcast')
snow = load_weather('snow')
num = min(len(overcast), len(snow))
overcast=overcast[0:num]
snow=snow[0:num]

input_shape = (3, 180, 320)
H = 180
W = 320

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class ImageDataset(Dataset):
    def __init__(self, A,B, transforms_ = None):
        self.transform = transforms.Compose(transforms_)
        self.A = A
        self.B = B

    def __getitem__(self, index):
        image_A = self.transform((self.A[index][:,:,0:3]).permute(2, 0, 1))/255
        image_B = self.transform((self.B[index][:,:,0:3]).permute(2, 0, 1))/255
        return {"A": image_A, "B": image_B}

    def __len__(self):
        return max(len(self.A), len(self.B))

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

def vramUsage():
	r = torch.cuda.memory_reserved(0)
	a = torch.cuda.memory_allocated(0)
	f = r-a  # free inside reserved
	return f


transforms_ = [
    transforms.Resize(int(H * 1.12), Image.BICUBIC),
    transforms.RandomCrop((H, W)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


dataLoader = DataLoader(ImageDataset(snow, overcast, transforms_=transforms_), 
                     batch_size = BATCH_SIZE, shuffle=True )


criterion_GAN = torch.nn.MSELoss().to(device)
criterion_cycle = torch.nn.L1Loss().to(device)
criterion_identity = torch.nn.L1Loss().to(device)

G_AB = Generator(input_shape,NUM_RESBLOCK)
G_BA = Generator(input_shape,NUM_RESBLOCK)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

G_AB.to(device)
G_BA.to(device)
D_A.to(device)
D_B.to(device)

optimizer_G = Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=BETA
)
optimizer_D_A = Adam(D_A.parameters(), lr=LR, betas=BETA)
optimizer_D_B = Adam(D_B.parameters(), lr=LR, betas=BETA)

# Learning rate update schedulers
lr_scheduler_G = lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(EPOCHS, 0, DECAY).step
)
lr_scheduler_D_A = lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(EPOCHS, 0, 100).step
)
lr_scheduler_D_B = lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(EPOCHS, 0, 100).step
)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

def log_images(real_A,real_B, step):
    """Saves a generated sample from the test set"""
    G_AB.eval()
    G_BA.eval()
    
    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)

    for i in fake_B:
        experiment.log_image( (i.permute(1, 2, 0)).detach().cpu(), name="fake snow", step = step)

    for i in fake_A:
        experiment.log_image( (i.permute(1, 2, 0)).detach().cpu(), name="fake overcast", step = step)

    del fake_B
    del fake_A
    torch.cuda.empty_cache()

    G_AB.train()
    G_BA.train()

print('start training')
with experiment.train():


    step = 0
    
    for epoch in range(EPOCHS):
        G_loss_val = 0
        D_loss_val = 0

        G_AB.train()
        G_BA.train()
        print("vram usage:", vramUsage())
        print("epochs:", epoch)

        for batch in dataLoader:
            
                real_A = batch["A"].to(device)
                real_B = batch["B"].to(device)
                valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)



                optimizer_G.zero_grad()

                # Identity loss
                loss_identity = (criterion_identity(G_BA(real_A), real_A) + criterion_identity(G_AB(real_B), real_B)) / 2

                # GAN loss
                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)

                loss_GAN = (criterion_GAN(D_B(fake_B), valid) + criterion_GAN(D_A(fake_A), valid)) / 2

                # Cycle loss
                recov_A = G_BA(fake_B)
                recov_B = G_AB(fake_A)

                loss_cycle = (criterion_cycle(recov_A, real_A) + criterion_cycle(recov_B, real_B)) / 2

                # Total loss
                loss_G = loss_GAN + 5 * loss_cycle + 10 * loss_identity
                G_loss_val += loss_G.item()
                loss_G.backward()
                optimizer_G.step()

                # -----------------------
                #  Train Discriminator A
                # -----------------------

                optimizer_D_A.zero_grad()

                # Real loss
                loss_real = criterion_GAN(D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
                # Total loss
                loss_D_A = (loss_real + loss_fake) / 2

                loss_D_A.backward()
                optimizer_D_A.step()

                # -----------------------
                #  Train Discriminator B
                # -----------------------

                optimizer_D_B.zero_grad()

                # Real loss
                loss_real = criterion_GAN(D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
                # Total loss
                loss_D_B = (loss_real + loss_fake) / 2

                loss_D_B.backward()
                optimizer_D_B.step()

                loss_D = (loss_D_A + loss_D_B) / 2
                D_loss_val += loss_D.item()
                # -----------------------
                #  Log
                # -----------------------
                if step % 500 == 0:
                    log_images(real_A, real_B, step)

                step += 1
            
                del real_A
                del real_B
                del loss_G
                del loss_D
                del loss_identity
                del loss_GAN
                del loss_cycle
                del valid
                del fake
                torch.cuda.empty_cache()

        experiment.log_metric("train_generator_loss", G_loss_val, epoch=epoch)  
        experiment.log_metric("train_discrminator_loss", D_loss_val, epoch=epoch)     

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        
        if epoch % 5 == 0:
            torch.save(G_AB.state_dict(), '/cluster/scratch/zhejiang/snow2overcast.model')
            torch.save(G_BA.state_dict(), '/cluster/scratch/zhejiang/overcast2snow.model')
            torch.save(D_A.state_dict(), '/cluster/scratch/zhejiang/D_snow.model')
            torch.save(D_B.state_dict(), '/cluster/scratch/zhejiang/D_overcast.model')
print('end training')