import torch
from model import ConstrastiveModel, Trainer
from vqa_data import Data_Creater
import argparse
import os
NOISE_MODES = ["none", "text", "image", "both"]
EPOCHS      = 5
BATCH_SIZE  = 64

def main(noise_mode, dupes, train):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    print(dupes, train)
    if train:
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
            trainer = Trainer(model, train_dl, test_dl, device)

            # 3) Train & save
            trainer.train(epochs=EPOCHS)
            if dupes:
                trainer.save_model(path=f"models/clip_{mode}_Dupes.pth")
            else:
                trainer.save_model(path=f"models/clip_{mode}_NoDupes.pth")
    else:
        dir="models/"
        for model in os.listdir(dir):
            path= os.path.join(dir, model)
            model.load_state_dict(torch.load(path))    
        
            correct = 0
            total = 0
            for batch_idx, (images, pos_emb, neg_emb) in enumerate(test_dl):
                # move everything to the device
                images = images.to(device)
                pos_emb = pos_emb.to(device)
                neg_emb = neg_emb.to(device)

                # get image embedding and text projection embeddings 
                with torch.no_grad():
                    images, pos_emb = model(images, pos_emb)
                    neg_emb = model.encode_descriptions(neg_emb)
                
                # Normalize each embedding to make consine sim equal to dot product
                images = F.normalize(images, p=2, dim=1)
                pos_emb = F.normalize(pos_emb, p=2, dim=1)
                neg_emb = F.normalize(neg_emb, p=2, dim=1)

                # go through each batch_size
                for i in range(images.shape[0]):
                    correct_cos_sim = torch.dot(images[i], pos_emb[i])
                    incorrect_cos_sim = torch.dot(images[i], neg_emb[i])

                    if correct_cos_sim > incorrect_cos_sim:
                        correct += 1
                    total += 1
            print(f"The accuracy of the model is {correct/total}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pick noise to add to dataset")

    parser.add_argument(
        "--mode", type=str, choices=NOISE_MODES, required=True,
        help=f"pick which input add noise to {NOISE_MODES}")
    
    parser.add_argument(
        "--dupes", type=str, choices=["True", "False"], required=True,
        help=f"Do you want to have duplicates? True or False?")
    
    parser.add_argument(
        "--train", type=str, choices=["True", "False"], required=True,
        help=f"Do you want to just train? True or False?")
    return parser

if __name__ == "__main__":
    args= build_parser().parse_args()
    main(args.mode, eval(args.dupes), eval(args.train))