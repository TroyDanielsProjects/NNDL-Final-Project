import torch
from model import ConstrastiveModel, Trainer
from vqa_data import Data_Creater
import argparse
import os
NOISE_MODES = ["none", "text", "image", "both"]
EPOCHS      = 5
BATCH_SIZE  = 64

def main(noise_mode, dupes):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    print(dupes)
    for mode in NOISE_MODES:
        print(f"\n\n=== Training with noise_mode = '{mode}' ===")
        # 1) Prepare data loaders for this regime
        creator = Data_Creater(noise_mode=mode)
        #specify dupes
        train_dl, val_dl, test_dl = creator.create_datasets(dupes)
        
        #debug: test datasets for noise
        #debug: sample=next(iter(train_dl))
        #debug:print(sample)

        # 2) Build model & trainer
        model   = ConstrastiveModel().to(device)
        trainer = Trainer(model, train_dl, val_dl, device)

        # 3) Train & save
        trainer.train(epochs=EPOCHS)
        if dupes:
            model_name=f"models/clip_{mode}_Dupes.pth"
        else:
            model_name=f"models/clip_{mode}_NoDupes.pth"
        trainer.test()
        trainer.save_model(path=model_name)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pick noise to add to dataset")

    parser.add_argument(
        "--noise", type=str, choices=NOISE_MODES, required=True,
        help=f"pick which input add noise to {NOISE_MODES}")
    
    parser.add_argument(
        "--dupes", type=str, choices=["True", "False"], required=True,
        help=f"Do you want to have duplicates? True or False?")
    
    return parser

if __name__ == "__main__":
    args= build_parser().parse_args()
    main(args.noise, eval(args.dupes))
