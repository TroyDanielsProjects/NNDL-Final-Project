import torch
from model import ConstrastiveModel, Trainer
from vqa_data import Data_Creater
import argparse
import os
import logging
from datetime import datetime
NOISE_MODES = ["none", "text", "image", "both"]
BATCH_SIZE  = 64


def main(noise_mode, dupes, epochs):
    # set up logging
    log_filename = f"logs/{datetime.now()}.log"
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    with open(log_filename, "a") as f:
        f.write(f"NEW RUN - Noise Mode:{noise_mode}, Dupes:{dupes}")
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='a'),
        logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Experiment started.")


    device = 'cuda' if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")
    logger.info(dupes)
    for mode in NOISE_MODES:
        logger.info(f"\n\n=== Training with noise_mode = '{mode}' ===")
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
        trainer.train(epochs=epochs)
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
        "--noise", type=str, choices=NOISE_MODES,
        help=f"pick which input add noise to {NOISE_MODES}",
        default="none")
    
    parser.add_argument(
        "--dupes", type=str, choices=["True", "False"],
        help=f"Do you want to have duplicates? True or False?",
        default="True")
    
    parser.add_argument(
        "--epochs", type=int,
        help=f"How many epochs do you want to run per experiment",
        default=15)
    
    return parser

if __name__ == "__main__":
    args= build_parser().parse_args()
    main(args.noise, eval(args.dupes), args.epochs)
