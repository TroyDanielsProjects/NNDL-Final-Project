from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms as T
from transformers import AutoTokenizer, AutoModel
import random
from nltk.corpus import wordnet
from torchvision.transforms.functional import to_pil_image
import hashlib
import logging
# --- TEXT AUGMENTATION HELPERS ---

logger = logging.getLogger(__name__)

def replace_with_synonym(word):
    """Replace a word with one of its WordNet synonyms (if available)."""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace('_', ' ')
            if candidate.lower() != word.lower():
                synonyms.append(candidate)
    return random.choice(synonyms) if synonyms else word

def random_char_perturb(word):
    """Randomly insert, delete, or swap a character in the given word."""
    if len(word) < 2:
        return word
    op = random.choice(['insert', 'delete', 'swap'])
    pos = random.randrange(len(word))
    if op == 'insert':
        letter = chr(random.randrange(ord('a'), ord('z')+1))
        return word[:pos] + letter + word[pos:]
    elif op == 'delete':
        return word[:pos] + word[pos+1:]
    else:  # swap
        if pos == len(word)-1:
            pos -= 1
        lst = list(word)
        lst[pos], lst[pos+1] = lst[pos+1], lst[pos]
        return ''.join(lst)

def augment_text(sentence):
    """Apply one synonym replacement and one char perturbation to the sentence."""
    words = sentence.split()
    # synonym replacement on one random long word
    candidates = [i for i,w in enumerate(words) if len(w)>3]
    
    #make sure the question prompt is not perturbed
    candidates.remove(0)
    if candidates:
        idx = random.choice(candidates)
        words[idx] = replace_with_synonym(words[idx])
    # char perturbation on another random long word
    candidates = [i for i,w in enumerate(words) if len(w)>3]
    if candidates:
        idx = random.choice(candidates)
        words[idx] = random_char_perturb(words[idx])
    return " ".join(words)

# --- DATASET CLASS ---

class VQA_Dataset(Dataset):
    # pass model and tokenizer here instead of creaing in constructor so that we avoid making multiple
    def __init__(self, split_ds, tokenizer, model, noise_mode="none", test=False):
        """
        split_ds    : HuggingFace Dataset split filtered to yes/no examples
        tokenizer   : a BERT tokenizer
        bert_model  : the corresponding BERT model
        noise_mode  : one of ["none","text","image","both"]
        """
        self.dataset = split_ds
        self.tokenizer = tokenizer
        self.model = model
        self.noise_mode = noise_mode
        self.test = test

        # Clean (baseline) image transform
        self.base_image_transform = T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std =[0.229,0.224,0.225]),
        ])
        # Noisy image transform (SimCLR‐style)
        self.image_noise_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15, fill=0),
            T.ColorJitter(brightness=0.4, contrast=0.4,
                            saturation=0.4, hue=0.1),
            T.GaussianBlur(kernel_size=5, sigma=(0.1,2.0)),
            T.Resize((256,256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std =[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        image = item['image']
        question = item['question']
        answer = item['answer']

        # 1) IMAGE: choose clean vs. noisy 
        if self.noise_mode in ("image", "both") and not self.test:
            img_tensor = self.image_noise_transform(image)
        else:
            img_tensor = self.base_image_transform(image)   

        # 2) TEXT: choose clean vs. noisy  ---> REVIEW 
        if self.noise_mode in ("text", "both") and not self.test:
            aug_q = augment_text(f"question: {question} answer: {answer}")
        else:
            aug_q = f"question: {question} answer: {answer}"

        # 3) BERT: tokenize and get embeddings
        tokenized = self.tokenizer(aug_q, padding='max_length', truncation=True, 
                                  max_length=100, return_tensors='pt')
        #commented out, caused error and was unused in subsequent lines
        #tokens = {k: v.squeeze(0) for k,v in tokens.items()}  # remove batch dim -> My addition, may not be needed
        with torch.no_grad():
                outputs = self.model(**tokenized)
        
        embedding = self.meanpooling(outputs, tokenized['attention_mask'])
        if self.test: # for test we need opposite answer for consine similarity. but don't want to take computational overhead for train
            if answer == "yes":
                 negative = "no"
            else:
                 negative = "yes"
            negative_prompt = f"question: {question} answer: {negative}"
            negative_tokenized = self.tokenizer(negative_prompt, padding='max_length', truncation=True, 
                                    max_length=100, return_tensors='pt')
            with torch.no_grad():
                    negative_outputs = self.model(**negative_tokenized)
            negative_embedding = self.meanpooling(negative_outputs, negative_tokenized['attention_mask'])

            return img_tensor, embedding.squeeze(0), negative_embedding.squeeze(0)
        else:
            return img_tensor, embedding.squeeze(0)
    
    #as per pubmedBERT docs:
    #Mean Pooling - Take attention mask into account for correct averaging
    #used to get single vector for entire sentence, instead of token-level vector embeddigns
    @staticmethod #doesnt pass self into meanpooling
    def meanpooling(output, mask):
        embeddings = output[0] # First element of model_output contains all token embeddings
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        

# --- DATA CREATER HELPER CLASS ---
# This class is used to create the datasets and dataloaders

class Data_Creater():
    def __init__(self, 
                 data_path = "flaviagiammarino/vqa-rad", 
                 model_path = "neuml/pubmedbert-base-embeddings",
                 noise_mode="none",
                 batch_size=64):
        self.vqa = load_dataset(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.noise_mode = noise_mode
        self.batch_size = batch_size
    
    @staticmethod
    def filter_duplicates(dataset):
        """
        Get idx of each unique image, only select these rows for dataset
        """
        image_list=[]
        unique_list=[]
        for idx, datapoint in enumerate(dataset):
            image=datapoint["image"]
            if image in image_list:
                continue
            else:
                image_list.append(image)
                unique_list.append(idx)
        return dataset.select(unique_list)

    def create_datasets(self, dupes):
        filter_lamda = lambda example: example["answer"] in ["yes", "no"]  # filter for yes/no answers
        logger.info(f"DUPES IS : {dupes}")
        if not dupes:
            self.vqa['train']=self.filter_duplicates(self.vqa['train'])
            self.vqa['test']=self.filter_duplicates(self.vqa['test'])
            #TEST THIS
        train_ds = self.vqa['train'].filter(filter_lamda)  # Create a filtered training dataset
        val_full_ds = self.vqa['test'].filter(filter_lamda)  # Create a filtered validation dataset, to be split later
        logger.info(f"NO DUPES: {len(train_ds)}")
        # Split the full validation dataset into 50% validation and 50% test
        split = val_full_ds.train_test_split(test_size=0.5, seed=42)
        val_ds    = split["train"]
        test_ds   = split["test"]

        # Wrap the datasets with the VQA_Dataset class
        train = VQA_Dataset(train_ds, self.tokenizer, self.model, noise_mode=self.noise_mode)
        validation = VQA_Dataset(val_ds, self.tokenizer, self.model, noise_mode=self.noise_mode, test=True)
        test = VQA_Dataset(test_ds, self.tokenizer, self.model, noise_mode=self.noise_mode, test=True)
        # Create dataloaders
        train_dataloader = self.create_dataloader(train)
        val_dataloader = self.create_dataloader(validation)
        test_dataloader = self.create_dataloader(test)

        return train_dataloader, val_dataloader, test_dataloader
    
    
    def create_dataloader(self, dataset, batch_size=64):
        return DataLoader(dataset, 
                          batch_size=batch_size, 
                          shuffle=True)
    
if __name__ == "__main__":
    dc = Data_Creater()
    train, val, test = dc.create_datasets(False)
    for batch_idx, (images, bert_encoding) in enumerate(train):
        logger.info(f"Batch Index: {batch_idx}")
        logger.info(f"Type of the image is {type(images[0])}")
        logger.info(f"Image is of shape: {images.shape}")
        logger.info(f"Type of the Embedding is {type(bert_encoding[0])}")
        logger.info(f"Encoding is of shape: {bert_encoding.shape}")
        break