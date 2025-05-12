import torch
from model import ConstrastiveModel, Trainer
from vqa_data import Data_Creater
import argparse

NOISE_MODES = ["none", "text", "image", "both"]
EPOCHS      = 5
BATCH_SIZE  = 64

def main(test_mode):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    for mode in NOISE_MODES:
        print(f"\n\n=== Training with noise_mode = '{mode}' ===")
        # 1) Prepare data loaders for this regime
        creator = Data_Creater(noise_mode=mode)
        train_dl, val_dl, test_dl = creator.create_datasets()
        
        #debug: test datasets for noise
        #debug: sample=next(iter(train_dl))
        #debug:print(sample)

        # 2) Build model & trainer
        model   = ConstrastiveModel().to(device)
        trainer = Trainer(model, train_dl, test_dl, device)

        # 3) Train & save
        trainer.train(epochs=EPOCHS)
        trainer.save_model(path=f"models/clip_{mode}.pth")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pick noise to add to dataset")

    parser.add_argument(
        "--mode", type=str, choices=NOISE_MODES, required=True,
        help=f"pick which input add noise to {NOISE_MODES}")
    return parser

if __name__ == "__main__":
    args= build_parser().parse_args()
    main(args.mode)