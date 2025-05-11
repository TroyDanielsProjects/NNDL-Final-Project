import torch
from model import ConstrastiveModel, Trainer
from vqa_data import Data_Creater

NOISE_MODES = ["none", "text", "image", "both"]
EPOCHS      = 5
BATCH_SIZE  = 64

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for mode in NOISE_MODES:
        print(f"\n\n=== Training with noise_mode = '{mode}' ===")
        # 1) Prepare data loaders for this regime
        creator = Data_Creater(noise_mode=mode)
        train_dl, val_dl, test_dl = creator.create_datasets()

        # 2) Build model & trainer
        model   = ConstrastiveModel().to(device)
        trainer = Trainer(model, train_dl, device)

        # 3) Train & save
        trainer.train(epochs=EPOCHS)
        trainer.save_model(path=f"models/clip_{mode}.pth")

if __name__ == "__main__":
    main()
