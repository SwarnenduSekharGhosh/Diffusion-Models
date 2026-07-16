import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Model import UNet
from Scheduler import GradualWarmupScheduler

def train(modelConfig : Dict): # indicates that modelConfig should be a dictionary. It is mainly for readability and type checking. Python doesnot strictly enforce it at runtime.
    device = torch.device(modelConfig["device"]) # this reads the device name from the configuration.
    # dataset creation
    dataset =CIFAR10(
        root = './CIFAR10', train=True, download=False,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # the first tuple represents the mean for each channel ; the second tuple represents the standard deviation
        ])) 
    # ToTensor() previously converted the values to [0,1], the final normalization converts 
    # them to approx [-1,1]
    dataloader = DataLoader(dataset, batch_size = modelConfig["batch_size"], 
                            shuffle = True, num_workers = 4, drop_last = True, pin_memory = True)
    # dataset is the dataset ; batch_szie determines how many images are processed in one optimization step ; 
    # shufflerandomly changes the order of the training examples at the beginning of every epoch
    # num_workers creates background workers, as the GPU trains on one batch the workers prepare upcoming batches.
    # drop_last : drops incomplete final batch so that every batch has equal number of training samples.
    
    # model setup
    net_model = UNet(T= modelConfig["T"], # total number of diffusion timesteps
                     ch=modelConfig["channel"],  # base number of feature channels in the U-Net
                     ch_mult=modelConfig["channel_mult"], # controls how the number of feature channels change at each U-net resolution
                     attn=modelConfig["attn"], # controls at which resolution levels attention blocks are used. This allows different spatial positions in a feature map to interact directly. 
                     num_res_blocks=modelConfig["num_res_blocks"], # controls the number of residual blocks used at each resolution level.
                     dropout=modelConfig["dropout"] # controls dropout inside the residual blocks. If dropout is 0.1 means 10% of selected activations will be set to zero during training.
                     ).to(device) # this moves the U-net parameters to the selected device.
    
    # Load existing weights only when provided
    if modelConfig["training_load_weight"] is not None: # this checks whether the user want to continue from the previously saved weights.
        checkpoint_path = os.path.join(      # constructing the check point path
             modelConfig["save_weight_dir"], 
             modelConfig["training_load_weight"]
        )
        net_model.load_state_dict( #copying the parameter values into the current U-Net.
             torch.load(checkpoint_path, map_location=device)
        )

        print(f"Loaded weights from:{checkpoint_path}") # this confirms by printing the current file that was loaded.
        
        #net_model.load_state_dict(
            #torch.load(os.path.join(
            #modelConfig["save_weight_dir"], 
            #modelConfig["training_load_weight"]), map_location=device))
        
    # Optimizer 
    optimizer = torch.optim.AdamW(
            net_model.parameters(),  # gives opitmizer acess to all trainable parameters of U-Net
            lr = modelConfig["lr"], #the learning rate controls the size of each parameter update.
              weight_decay=1e-4) # weight decay discourages model parameters from becoming unnecessarily large.
    # the optimizer updates the U-Net parameters after backpropagation
    # "AdamW" applies weight decay separately from the adaptive gradient update, unlike the original "Adam" implementation with ordinary L2 regularization.
    # Learning-rate schedulers
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
         # this scehduler gradually lowers the learning rate following a cosine-shaped curve.
            optimizer=optimizer, 
            T_max = modelConfig["epoch"], # controls the length of the cosine schedule
            eta_min=0,last_epoch= -1) # eta_min = 0 is the minimum learning rate, last_epoch = -1 tells the pytorch that training is beginning from the initial sceduler state.
    warmUpScheduler = GradualWarmupScheduler( # warmup gradually increases the learning rate at the beginning of training.
            optimizer = optimizer, # this warmup can prevent unstable updates during the first few epochs when the network is randomly initialized. 
            multiplier= modelConfig["multiplier"], # This controls the target learning rate relative to the optimizer's initial learning rate.
            warm_epoch= modelConfig["epoch"] // 10, # how many epochs of the total number of epochs will be used as warm-up after that the cosine scheduler takes over.
            after_scheduler= cosineScheduler)
    
    # Diffusion trainer
    # This object wraps the U-Net and implements the forward diffusion training procedure
    trainer = GaussianDiffusionTrainer(
            net_model, 
            modelConfig["beta_1"], # This is the starting variance of the noise schedule.
            modelConfig["beta_T"], # This is the final variance in the noise schedule.
            modelConfig["T"]  #  This is the number of diffusion steps.
            ).to(device)
    
    """
    1. Receive a clean image x_0
    2. Randomly choose a timestep t.
    3. Sample Gaussian noise ϵ.
    4. Construct the noisy image x_t
    5. Pass x_t and t through the U-Net.
    6. Compare the predicted noise against the true noise.
    """
    
    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
    # This creates the directory where weights will be saved.

    # start training
    for e in range(modelConfig["epoch"]): # This repeats training for requested number of epochs.
        net_model.train() # this puts the U-net into training mode

        with tqdm(dataloader, # creating the progress bar
                  dynamic_ncols=True
                  ) as tqdmDataLoader:
                # iterating through batches
                for images, _ in tqdmDataLoader: # CIFAR-10 returns two items for every batch images, labels
                                                 # '_' is used as a standard DDPM learns the image distribution without using class labels                    
                    # train
                    optimizer.zero_grad()
                    # pytorch accumulates gradients by default
                    # so before calculating gradients for each new batch, we reset them to zero
                    x_0 = images.to(device, non_blocking = True)

                    loss = trainer(x_0).sum() / 1000.
                    # the trainer(x_0) calls the forward() method of GaussianDiffusionTrainer.
                    # it returns an element-wise squared error tensor. and then sums every loss value across the batch, channels, image height, image width.
                    # then scale the sum by 1000

                    # loss = trainer(x_0).mean() #For batch size B, channels C, height H, and width W, this will divide by B X C X H X W
                    
                    loss.backward() # this performs backpropagation
                    
                    torch.nn.utils.clip_grad_norm_(
                        net_model.parameters(), 
                        modelConfig["grad_clip"]
                    )
                    
                    optimizer.step()
                    
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch" : e #+ 1,
                        "loss: ": loss.item(),
                        "img shape: ": x_0.shape,
                        #"LR": optimizer.state_dict()['para_groups'][0]["lr"]
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                         }
                    )
        # Update learning rate once after each epoch
        warmUpScheduler.step()

        # save checkpoint
        checkpoint_path = os.path.join(
             modelConfig["save_weight_dir"],
             f"ckpt_{e+1}.pt"
        )

        torch.save(net_model.state_dict(), checkpoint_path)
                             