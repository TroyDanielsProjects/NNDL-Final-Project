from datasets import load_dataset, DatasetDict, Dataset, Image
from transformers import AutoTokenizer, AutoModel
import torch
from torchvision import transforms 
from torch.utils.data import DataLoader

class VQA_Dataset():

    def __init__(self):
        """
        Initialize VQA Dataset.
        Note JpegImageFile is unhashable --> cannot use a set
        """
        self.vqa= load_dataset("flaviagiammarino/vqa-rad")
        self.yes_no_train=0
        self.yes_no_test=0
        self.total_train=len(self.vqa["train"])
        self.total_test=len(self.vqa["test"])
        self.vqa_yn= DatasetDict()
        self.train= DatasetDict()
        self.val= DatasetDict()
        self.test= DatasetDict()
        self.image_set=[]
        self.torches={}

    #turn PIL into a tensor 
    @staticmethod
    def tensorize_image(image):
        image_preprocessing = transforms.Compose([
            #want constant size
            transforms.Resize((256,256)),
            #use ToTensor gets float32 dtype between [0,1]
            #PILToTensor gets uint8 between [0,255]
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # this mean and std are one of the industry standards
        ])
        transformed_image = image_preprocessing(image)
        return transformed_image

    def get_stats(self):
        for point in self.vqa["train"]:
            answer=point["answer"]
            if answer == "yes" or answer == "no":
                self.yes_no_train+=1
        print(f"yes no questions out of all train questions: {self.yes_no_train}/{self.total_train}")

        for point in self.vqa["test"]:
            answer=point["answer"]
            if answer == "yes" or answer == "no":
                self.yes_no_test+=1
        print(f"yes no questions out of all test questions: {self.yes_no_test}/{self.total_test}")
    
    def get_prompt_dataset(self):
        #train, test
        for split in self.vqa.keys():
            new_data = []
            for point in self.vqa[split]:
                #only want yes/no answers
                if point["answer"] not in ["yes", "no"]:
                    continue
                #skip duplicate images
                '''
                if point["image"] in self.image_set:
                    continue
                else:
                    self.image_set.append(point["image"])
                '''
                #item format: "image: ____ , question: ____ anwser:_____."
                item = {
                    "image": self.tensorize_image(point["image"]),
                    "prompt": f"question: {point['question']} answer: {point['answer']}"
                }
                new_data.append(item)
            self.vqa_yn[split] = Dataset.from_list(new_data)

        #now split prepped data into neccessary test/val/train
        #take 10% of training data and use for validation set, use seed for reproducability
        train_val = self.vqa_yn["train"].train_test_split(test_size=0.1, seed=42)
        self.train = train_val["train"]
        self.val = train_val["test"]
        self.test = self.vqa_yn["test"]
        self.dataset = DatasetDict({
            "train": self.train,
            "validation": self.val,
            "test": self.test
        })
        print(self.dataset)
        return self.vqa, self.dataset

    def get_torch(self, batch_size=4):
        for split in self.dataset.keys():
            ds = self.dataset[split].with_format("torch")
            self.torches[split] = DataLoader(ds, batch_size)
            breakpoint()
        return self.torches

class PubMedBERT():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
        self.model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")
        self.sentences= {}
        self.embeddings={}
    
    #as per pubmedBERT docs:
    #Mean Pooling - Take attention mask into account for correct averaging
    #used to get single vector for entire sentence, instead of token-level vector embeddigns
    @staticmethod #doesnt pass self into meanpooling
    def meanpooling(output, mask):
        embeddings = output[0] # First element of model_output contains all token embeddings
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    
    #Sentences we want sentence embeddings for
    def get_sentences(self, dataset):
        for split in dataset.keys():
            sentences=[]
            for point in dataset[split]:
                sentences.append(point['prompt'])
            self.sentences[split]=sentences

    #Compute token embeddings using padding, masks and mean pooling
    def get_embeddings(self, dataset):
        #get inputs, outputs and embedding for each train/val/test
        for split in dataset.keys():
            inputs= self.tokenizer(self.sentences[split], padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
            #Perform pooling. In this case, mean pooling.
            embedding = self.meanpooling(outputs, inputs['attention_mask'])
            self.embeddings[split]=embedding


        print("Sentence embeddings:")
        print(self.embeddings)
        train=self.embeddings['train']
        #want BERT 768 second dimension
        print(train.shape)
        #no NaN or exploding numbers
        print(torch.min(train), torch.max(train))
        return self.embeddings
   

if __name__ == "__main__":
    dataset=VQA_Dataset()
    dataset.get_stats()
    vqa, cleaned_vqa = dataset.get_prompt_dataset()
    breakpoint()
    vqa_dataloader= dataset.get_torch(batch_size=4)

    #dataloader plugin
    pubmedbert= PubMedBERT()
    pubmedbert.get_sentences(cleaned_vqa)
    pubmedbert.get_embeddings(cleaned_vqa)