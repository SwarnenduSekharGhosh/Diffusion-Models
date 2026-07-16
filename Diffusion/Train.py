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
                     ).to(device)
    
    # Load existing weights only when provided
    if modelConfig["training_load_weight"] is not None:
        checkpoint_path = os.path.join(
             modelConfig["save_weight_dir"],
             modelConfig["training_load_weight"]
        )
        net_model.load_state_dict(
             torch.load(checkpoint_path, map_location=device)
        )

        print(f"Loaded weights from:{checkpoint_path}")

        #net_model.load_state_dict(
            #torch.load(os.path.join(
            #modelConfig["save_weight_dir"], 
            #modelConfig["training_load_weight"]), map_location=device))
        
    # Optimizer 
    optimizer = torch.optim.AdamW(
            net_model.parameters(), 
            lr = modelConfig["lr"],
              weight_decay=1e-4)
    
    # Learning-rate schedulers
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            T_max = modelConfig["epoch"],
            eta_min=0,last_epoch= -1)
    warmUpScheduler = GradualWarmupScheduler(
            optimizer = optimizer, 
            multiplier=modelConfig["multiplier"], 
            warm_epoch= modelConfig["epoch"] // 10, 
            after_scheduler=cosineScheduler)
    
    # Diffusion trainer
    trainer = GaussianDiffusionTrainer(
            net_model, 
            modelConfig["beta_1"],
            modelConfig["beta_T"],
            modelConfig["T"]
            ).to(device)
    
    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)

    # start training
    for e in range(modelConfig["epoch"]):
        net_model.train()

        with tqdm(dataloader, 
                  dynamic_ncols=True
                  ) as tqdmDataLoader:
                for images, _ in tqdmDataLoader:
                    # train
                    optimizer.zero_grad()

                    x_0 = images.to(device, non_blocking = True)

                    loss = trainer(x_0).sum() / 1000.
                    # loss = trainer(x_0).mean()
                    
                    loss.backward()
                    
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
                             