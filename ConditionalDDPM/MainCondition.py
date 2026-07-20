from TrainCondition import train, evaluate


def main(model_config=None):
    modelConfig = {
        "state": "train", # train or evaluate
        "epoch": 70,
        "batch_size": 80,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "D:/DL_Projects/Diffusion-Models_checkpointfolder/DiffusionConditional/Checkpoints/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_200.pt",
        "sampled_dir": "D:/DL_Projects/Diffusion-Models_checkpointfolder/DiffusionConditional/SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()