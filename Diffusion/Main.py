from Train import train, evaluate


def main(model_config = None):
    modelConfig = {
        "state": "train", # or evaluate
        # Diffusion settings
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        # Unet settings
        "channel": 128,
        "channel_mult": [1, 2, 3, 4], #128,256,384,512
        "attn": [2], # attention is applied at level 2
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "D:/DL_Projects/Diffusion-Models_checkpointfolder/Diffusion/Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "D:/DL_Projects/Diffusion-Models_checkpointfolder/Diffusion/SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig["state"] == "evaluate":
        evaluate(modelConfig)

    else:
        raise ValueError("modelConfig['state'] must be either 'train' or 'evaluate'.")
    


if __name__ == '__main__':
    main()