from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel

class VQA_Dataset(Dataset):

    # pass model and tokenizer here instead of creaing in constructor so that we avoid making multiple
    def __init__(self, ds, transformations, tokenizer, model, test=False):
        self.dataset = ds
        self.transformations = transformations
        self.tokenizer = tokenizer
        self.model = model
        self.test = test

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        image = item['image']
        question = item['question']
        answer = item['answer']
        if self.transformations:
            image = self.transformations(image)
        prompt = f"question: {question} answer: {answer}"
        tokenized = self.tokenizer(prompt, padding='max_length', truncation=True, 
                                  max_length=100, return_tensors='pt')
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

            return image, embedding.squeeze(0), negative_embedding.squeeze(0)
        else:
            return image, embedding.squeeze(0)
    
    #as per pubmedBERT docs:
    #Mean Pooling - Take attention mask into account for correct averaging
    #used to get single vector for entire sentence, instead of token-level vector embeddigns
    @staticmethod #doesnt pass self into meanpooling
    def meanpooling(output, mask):
        embeddings = output[0] # First element of model_output contains all token embeddings
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        

class Data_Creater():

    def __init__(self, data_path = "flaviagiammarino/vqa-rad", model_path = "neuml/pubmedbert-base-embeddings"):
        self.vqa = load_dataset(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

        self.image_preprocessing = transforms.Compose([
            #want constant size
            transforms.Resize((256,256)),
            #use ToTensor gets float32 dtype between [0,1]
            #PILToTensor gets uint8 between [0,255]
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # this mean and std are one of the industry standards
        ])
    
    def create_datasets(self):
        filter_lamda = lambda example: example["answer"] in ["yes", "no"]
        train = VQA_Dataset(self.vqa['train'].filter(filter_lamda), self.image_preprocessing, self.tokenizer, self.model)
        test = self.vqa['test'].filter(filter_lamda)
        test = VQA_Dataset(test, self.image_preprocessing, self.tokenizer, self.model, test=True)

        train_dataloader = self.create_dataloader(train)
        test_dataloader = self.create_dataloader(test)

        return train_dataloader, test_dataloader
    
    
    def create_dataloader(self, dataset, batch_size=64):
        return DataLoader(dataset, 
                          batch_size=batch_size, 
                          shuffle=True)
    
if __name__ == "__main__":
    dc = Data_Creater()
    train, val, test = dc.create_datasets()
    for batch_idx, (images, bert_encoding) in enumerate(train):
        print(f"Batch Index: {batch_idx}")
        print(f"Type of the image is {type(images[0])}")
        print(f"Image is of shape: {images.shape}")
        print(f"Type of the Embedding is {type(bert_encoding[0])}")
        print(f"Encoding is of shape: {bert_encoding.shape}")
        break
    
