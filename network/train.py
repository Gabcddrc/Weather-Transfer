from comet_ml import Experiment
import torch
import sys
sys.path.append('/psi/home/li_s1/DeepLearning2021')
from utils.bdd100k_dataloader import *
from utils.helper import *
from generator import *
from discriminator import Discriminator2
from torch.optim import Adam, lr_scheduler
import random
from utils.config import Config
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

# +
config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))
print(f"We have {torch.cuda.device_count()} devices available")
#CKPT_PATH="/cluster/scratch/zhejiang/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
#CKPT_PATH="/psi/home/li_s1/data/Season/pretrained/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
CKPT_PATH="/psi/home/li_s1/data/Season/pretrained/latest_deeplabv3plus_mobilenet_BDD100k_os16.pth"
MODE = sys.argv[1]

jobname = sys.argv[2]
import os
if not os.path.isdir(f'/psi/home/li_s1/data/Season/training/{jobname}'):
    os.mkdir(f'/psi/home/li_s1/data/Season/training/{jobname}')

BATCH_SIZE = int(sys.argv[3])
EPOCHS = 200
LR = 2e-4
# convex combination between semantic, similariy and realism loss. sum should be in [0, 1]
LAMBDA_SEM = 5
LAMBDA_REAL = 2
LAMBDA_SIM = 10
# CLIP GRADIENT
MAX_GRADIENT_G = 1.0
MAX_GRADIENT_D = 1.0
# learning rate decay
DECAY_G = 10
DECAY_D = 10
# skip discriminator with prob
SKIP_D = 0.2
# PARAMS FOR OPTIMIZER
BETA = (0.5, 0.999)
# LABELS
REAL_LABEL = 1.0
FAKE_LABEL = 0.0
# the network transforms A -> B
DOMAIN_A = "sunny"
DOMAIN_B = "snowy"
STEP_LOG_IMAGE = 200
# -

experiment = Experiment(
    api_key="HyWl3zINDt5tVxrEffWKpJ0Ms",
    project_name="deep-learning",
    workspace="jjiang",
)

#experiment.set_name('training-merlinGPU')
experiment.set_name(jobname)
# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "current_learning_rate": LR,
    "model": 'our_GAN',
    "EPOCHS": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lambda_real": LAMBDA_REAL,
    "lambda_sem": LAMBDA_SEM,
    "lambda_sim": LAMBDA_SIM,
    "decay_g": DECAY_G,
    "decay_d": DECAY_D,
    "skip_d": SKIP_D,
    "beta": BETA
}

experiment.log_parameters(hyper_params)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


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
# class ImageDataset(Dataset):
#     def __init__(self, A,B):
#         self.A = A
#         self.B = B

#     def __getitem__(self, index):
#         image_A = self.A[index]
#         image_B = self.B[index]
#         return {"A": image_A, "B": image_B}

#     def __len__(self):
#         return min(len(self.A), len(self.B))

# DATALOADER

# adjust the input shape to the correct one
# 3 rgb + 19/21 semantic channels
input_shape = (BATCH_SIZE, 22, 224, 400)
print('loading datasets')
# Data sets
# generator
dataset_train_A = Bdd100kDataset(config, mode=MODE, weather='overcast', keep_in_memory=False)
#dataset_train_A = Bdd100kDataset(config, mode='val', weather='overcast', keep_in_memory=False)
# discriminator
dataset_train_B = Bdd100kDataset(config, mode=MODE, weather='snowy', keep_in_memory=False)
#dataset_train_B = Bdd100kDataset(config, mode='val', weather='snowy', keep_in_memory=False)


# dataLoader = DataLoader(ImageDataset(dataset_train_A, dataset_train_B), 
#                      batch_size = BATCH_SIZE, shuffle=True )

# Data loaders
dataloader_A_train = DataLoader(dataset_train_A, batch_size=BATCH_SIZE, shuffle=True, num_workers=64, prefetch_factor=4)
dataloader_B_train = DataLoader(dataset_train_B, batch_size=BATCH_SIZE, shuffle=True, num_workers=64, prefetch_factor=4)
# dataloader_A_val = DataLoader(dataset_val_A, batch_size=BATCH_SIZE, shuffle=True)
# dataloader_B_val = DataLoader(dataset_val_B, batch_size=BATCH_SIZE, shuffle=True)

print('Finishing datasets')


# NETWORK CREATION
# TODO fine tune the hyperparameters to achieve a reasonable amount
generator = Generator(in_channels_transl=input_shape[1],
                              block_sizes_transl=[64, 128, 256],
                              kernel_size_transl=3,
                              kernel_size_upsampl_transl=3,
                              input_shape_attention=input_shape,
                              attention_module_output_channels=1,
                              kernel_size_blending=(3, 3),
                              kernel_size_final_filter=(3, 3),
                              multi_output=True)
discriminator = Discriminator2(CKPT_PATH, (3,224,400))

# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

generator.load_state_dict(torch.load(f'/psi/home/li_s1/data/Season/training/a100_long/gan.model'))
discriminator.load_state_dict(torch.load(f'/psi/home/li_s1/data/Season/training/a100_long/discriminator.model'))

if jobname.startswith('a100'):
    generator= nn.DataParallel(generator, device_ids = [0,1,2,3])
    discriminator= nn.DataParallel(discriminator, device_ids = [1,0,2,3])
    generator.to(f'cuda:{generator.device_ids[0]}')
    discriminator.to(f'cuda:{discriminator.device_ids[0]}')
else:
    generator= nn.DataParallel(generator)
    discriminator= nn.DataParallel(discriminator)
    generator.to(device)
    discriminator.to(device)


# OPTIMIZER CREATION
optimizer_G = Adam(
    generator.module.parameters(), lr=LR, betas=BETA
)

# TODO check if the parameters need to be limited to the ones which were set to
# requires_grad = True, we dont want to retrain resnet or Deeplab
optimizer_D = Adam(discriminator.module.parameters(), lr=LR, betas=BETA)

# Learning rate update schedulers
lr_scheduler_G = lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(EPOCHS, 0, DECAY_G).step
)
lr_scheduler_D = lr_scheduler.LambdaLR(
    optimizer_D, lr_lambda=LambdaLR(EPOCHS, 0, DECAY_D).step
)

# LOSSES
# TODO check that this is the correct loss for semantic segmentation
semantic_consistency_loss = torch.nn.L1Loss().to(device)

# TODO check that the binary cross entropy loss is correct for 1x2 tensors (real/fake)
realism_loss = torch.nn.MSELoss().to(device)

similarity_loss = torch.nn.L1Loss().to(device)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


def log_images(real, before_map, after_map, step):
    """Saves a generated sample from the test set"""
    #need to re-run the generator on the image, because we want it to be in eval mode
    generator.module.eval()
    fake, attention, translation, pointwise = generator.module(images_A_train)
    
    for i in range(BATCH_SIZE):
        fig, ax = plt.subplots(2, 2, figsize=(40, 22.4))
        ax[0, 0].imshow(real[i].permute(1, 2, 0)[:,:,:3].detach().cpu())
        ax[1, 0].imshow(fake[i].permute(1, 2, 0)[:,:,:3].detach().cpu())
        #how to display segmentation map depends on the actual data 
        
        ax[0, 1].imshow(np.argmax(before_map[i].detach().cpu(), axis=0))
        ax[1, 1].imshow(np.argmax(after_map[i].detach().cpu(), axis=0))
        img = fig2img(fig)
        experiment.log_image(img, step=step)
        
        experiment.log_image( (attention[i].permute(1, 2, 0)).detach().cpu(), name="attention", step = step)
        experiment.log_image( (translation[i].permute(1, 2, 0)).detach().cpu(), name="translation", step = step)
        experiment.log_image( (pointwise[i].permute(1, 2, 0)).detach().cpu(), name="pointwise", step = step)
    generator.module.train()

fake_B_buffer = ReplayBuffer()

print('start training')
with experiment.train():
    step = 0

    for epoch in range(EPOCHS):

        generator.module.train()
        print("epochs:", epoch)
        losses = {
            "d_real": [],
            "d_fake": [],
            "d_total": [],
            "g_real": [],
            "g_sem": [],
            "g_sim": [],
            "g_total": [],
        }
        domain_B_iterator = iter(dataloader_B_train)
        # TODO we want to go through one epoch, sample the maps and images from domain A and B
        for batch_idx, images_A_train in enumerate(dataloader_A_train):

            # here we get the next batch of domain B. This way one epoch is defined by one
            # iteration through A, but we can utilize all data from B
            # see https://stackoverflow.com/questions/51444059/how-to-iterate-over-two-dataloaders-simultaneously-using-pytorch
            try:
                images_B_train = next(domain_B_iterator)
            except StopIteration:
                domain_B_iterator = iter(dataloader_B_train)
                images_B_train = next(domain_B_iterator)

            if images_A_train.size(0) != images_B_train.size(0):
                print("Error batch sizes are not the same")
                continue

            # load both images
            # real_data should be shape [b, 19 or 21, H, W]
            # the discriminator only taks [b, 3, H, W] hence slicing is needed
            images_A_train = images_A_train.to(device)
            images_B_train = images_B_train.to(device)
            valid = Variable(Tensor(np.ones((images_A_train.size(0), *discriminator.module.output_shape))), requires_grad=False)
            valid_smooth = Variable(Tensor(np.full(shape=(images_A_train.size(0), *discriminator.module.output_shape), fill_value=0.9)), requires_grad=False)

            fake = Variable(Tensor(np.zeros((images_A_train.size(0), *discriminator.module.output_shape))), requires_grad=False)
            fake_smooth = Variable(Tensor(np.full(shape=(images_A_train.size(0), *discriminator.module.output_shape), fill_value=0.1)), requires_grad=False)

            ######################
            # train the generator
            ######################
            generator.module.zero_grad()
            # be as close to the real label as possible
            fake_images = generator.module(images_A_train)[0]
            map_after, pred = discriminator.module(fake_images)

            # the map before should not influence anything
            map_before = images_A_train[:, 3:, :, :]

            # compute losses
            g_realism_loss = realism_loss(pred, valid)
            g_semantic_loss = semantic_consistency_loss(map_after, map_before)
            g_similarity_loss = similarity_loss(generator.module(images_B_train)[0], images_B_train[:, :3, :, :])

            # add and backprog the losses
            g_total_loss = LAMBDA_REAL * g_realism_loss + \
                           LAMBDA_SEM * g_semantic_loss + \
                           LAMBDA_SIM * g_similarity_loss
            losses["g_real"].append(g_realism_loss.item())
            losses["g_sem"].append(g_semantic_loss.item())
            losses["g_sim"].append(g_similarity_loss.item())
            losses["g_total"].append(g_total_loss.item())
            # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=MAX_GRADIENT_G)
            g_total_loss.backward()
 
            optimizer_G.step()



            discriminator.module.zero_grad()
            # train discriminator with prob 1-SKIP_D
            if random.random() > SKIP_D:
                ########################
                # Prediction on all real
                ########################
                # dont need map
                _, pred = discriminator.module(images_B_train[:, :3, :, :].detach())
                # compute the loss
                d_loss_real = realism_loss(pred, valid_smooth)
                # propagate that error back

                #######################
                # compute all fake loss
                #######################

                fake_images = generator.module(images_A_train)[0]

                fake_B = fake_B_buffer.push_and_pop(fake_images)
                # print(fake_images.shape)
                # fill tensor in place
                
                # classify without loss through generator, dont need sem map
                _, pred = discriminator.module(fake_B.detach())

                d_loss_fake = realism_loss(pred, fake_smooth)

                errD = (d_loss_real + d_loss_fake)/2
                losses["d_real"].append(d_loss_real.item())
                losses["d_fake"].append(d_loss_fake.item())
                losses["d_total"].append(errD.item())

                # clip the gradient
                # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=MAX_GRADIENT_D)

                errD.backward()
                optimizer_D.step()



            if step % STEP_LOG_IMAGE == 0:
                # log some images
                log_images(images_A_train, map_before, map_after, step)

            # if step % 400 == 0:
            #     # go through validation and report some images and discriminator loss
            #     pass

            step += 1
         

        # TODO should that one be outside the train loop?
        lr_scheduler_D.step()
        lr_scheduler_G.step()

        # TODO check that these losses are even correct i am very unsure
        experiment.log_metric("train_generator_real_loss", np.sum(losses["g_real"]), epoch=epoch)
        experiment.log_metric("train_generator_sem_loss", np.sum(losses["g_sem"]), epoch=epoch)
        experiment.log_metric("train_generator_sim_loss", np.sum(losses["g_sim"]), epoch=epoch)
        experiment.log_metric("train_generator_total_loss", np.sum(losses["g_total"]), epoch=epoch)

        experiment.log_metric("train_discriminator_real_loss", np.sum(losses["d_real"]),
                              epoch=epoch)
        experiment.log_metric("train_discriminator_fake_loss", np.sum(losses["d_fake"]), epoch=epoch)
        experiment.log_metric("train_discriminator_total_loss", np.sum(losses["d_total"]),
                              epoch=epoch)

        experiment.log_metric("current_learning_rate generator", lr_scheduler_G.get_last_lr(),
                              epoch=epoch)
        experiment.log_metric("current_learning_rate discriminator", lr_scheduler_D.get_last_lr(),
                              epoch=epoch)

        if epoch % 5 == 0:
            torch.save(generator.module.state_dict(), f'/psi/home/li_s1/data/Season/training/{jobname}/gan.model')
            torch.save(discriminator.module.state_dict(), f'/psi/home/li_s1/data/Season/training/{jobname}/discriminator.model')
            pass
