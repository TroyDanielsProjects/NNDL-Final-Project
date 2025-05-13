import torch
from model import ConstrastiveModel, Trainer
from vqa_data import Data_Creater
import argparse
import os
import logging
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def test_model(dupes, epochs, batch_size, path):

     # set up logging
    log_filename = f"logs/{datetime.now()}.log"
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    with open(log_filename, "a") as f:
        f.write(f"NEW RUN - Dupes:{dupes}, Batch Size: {batch_size}")
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='a'),
        logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Testing started.")
    
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    device = 'mps'
    logger.info(f"Using {device} device")
    # 1) Prepare data loaders for this regime
    creator = Data_Creater(batch_size=batch_size)
    #specify dupes
    train_dl, val_dl, test_dl = creator.create_datasets(dupes)

    # 2) Build model & trainer
    model   = ConstrastiveModel().to(device)
    trainer = Trainer(model, train_dl, val_dl, device)
    trainer.load_model(path)
    trainer.test()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pick noise to add to dataset")
    
    parser.add_argument(
        "--dupes", type=str, choices=["True", "False"],
        help=f"Do you want to have duplicates? True or False?",
        default="True")
    
    parser.add_argument(
        "--epochs", type=int,
        help=f"How many epochs do you want to run per experiment",
        default=15)
    
    parser.add_argument(
        "--batch_size", type=int,
        help=f"Set the batch size for each experiment",
        default=64)
    
    parser.add_argument(
        "--model_path", type=str,
        help=f"Set the batch size for each experiment",
        default="./models/clip_none_Dupes.pth")
    
    return parser

if __name__ == "__main__":
    args= build_parser().parse_args()
    test_model(eval(args.dupes), args.epochs, args.batch_size, args.model_path)
