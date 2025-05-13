import torch
from model import ConstrastiveModel, Trainer
from vqa_data import Data_Creater
import argparse
import os
import logging
from datetime import datetime
NOISE_MODES = ["none", "text", "image", "both"]



def main(dupes, epochs, batch_size, repeats):
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
    logger.info("Experiment started.")


    device = 'cuda' if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")
    logger.info(dupes)
    total_accuracy_difference_none = 0
    total_accuracy_difference_text = 0
    total_accuracy_difference_image = 0
    total_accuracy_difference_both = 0
    for i in repeats:
        logger.info(f"Starting iteration {i+1}")
        for mode in NOISE_MODES:
            logger.info(f"\n\n=== Training with noise_mode = '{mode}' ===")
            # 1) Prepare data loaders for this regime
            creator = Data_Creater(noise_mode=mode, batch_size=batch_size)
            #specify dupes
            train_dl, val_dl, test_dl = creator.create_datasets(dupes)
            
            #debug: test datasets for noise
            #debug: sample=next(iter(train_dl))
            #debug:print(sample)

            # 2) Build model & trainer
            model   = ConstrastiveModel().to(device)
            trainer = Trainer(model, train_dl, val_dl, device)

            # 3) Train & save
            start_accuracy = trainer.test()
            trainer.train(epochs=epochs)
            end_accuracy = trainer.test()
            diff = end_accuracy - start_accuracy
            if mode == 'none':
                total_accuracy_difference_none+=diff
                logger.info(f"For {mode} - difference in train-original:{diff}, total={total_accuracy_difference_none}")
            elif mode == 'text':
                total_accuracy_difference_text+=diff
                logger.info(f"For {mode} - difference in train-original:{diff}, total={total_accuracy_difference_text}")
            elif mode == 'image':
                total_accuracy_difference_image+=diff
                logger.info(f"For {mode} - difference in train-original:{diff}, total={total_accuracy_difference_image}")
            else:
                total_accuracy_difference_both+=diff
                logger.info(f"For {mode} - difference in train-original:{diff}, total={total_accuracy_difference_both}")

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
        default=20)
    
    parser.add_argument(
        "--batch_size", type=int,
        help=f"Set the batch size for each experiment",
        default=64)
    
    parser.add_argument(
        "--repeats", type=int,
        help=f"Set the batch size for each experiment",
        default=10)
    
    return parser

if __name__ == "__main__":
    args= build_parser().parse_args()
    main(eval(args.dupes), args.epochs, args.batch_size, args.repeats)
