

import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from ModelCondition import UNet
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root=r"D:\Deep_Learning\Data\GenAI\CIFAR10", train=True, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, 
        num_workers=4, 
        drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], 
                     ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], 
                     dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), 
            map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, 
                                             multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, 
                                             after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) + 1 # this model reserves label 0 for no condition.
                # 0 = null / unconditional label ; 1–10 = real CIFAR classes
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                # With 10% probability, replace the real class labels with label 0.    
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        
        # torch.save(net_model.state_dict(), os.path.join(
        #    modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))
        
        if (e + 1) % 50 == 0:
            checkpoint_path = os.path.join(
                modelConfig["save_dir"],
                f"ckpt_{e+1}.pt"
            )

            torch.save(net_model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


"""
CIFAR image x₀
CIFAR label y
        ↓
shift label: y + 1
        ↓
sometimes replace y with 0
        ↓
choose random timestep t
        ↓
add noise to x₀ → x_t
        ↓
U-Net receives x_t, t, y
        ↓
time embedding tells noise level
class embedding tells desired class
        ↓
U-Net predicts noise
        ↓
MSE with true noise
        ↓
update model
"""            

def evaluate(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # load model and evaluate
    with torch.no_grad():
        step = int(modelConfig["batch_size"] // 10)
        labelList = []
        k = 0
        for i in range(1, modelConfig["batch_size"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).long().to(device) + 1

        print("labels: ", labels)

        model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], 
                     ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], 
                     dropout=modelConfig["dropout"]).to(device)
        
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), 
            map_location=device)
        
        model.load_state_dict(ckpt)

        print("model load weight done.")

        model.eval()

        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], 
            modelConfig["T"], 
            w=modelConfig["w"]).to(device)
        
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], 
                  modelConfig["img_size"]], device=device)
        
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)

        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), 
            nrow=modelConfig["nrow"])
        
        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]

        print(sampledImgs)

        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), 
            nrow=modelConfig["nrow"])
        
"""
Choose labels y
Create random noise x_T
        ↓
For t = T-1 down to 0:
        ↓
predict conditional noise using label y
predict unconditional noise using label 0
combine them using guidance strength w
compute x_{t-1}
        ↓
Final x₀ is generated image

"""        