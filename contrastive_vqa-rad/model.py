from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel
from torchvision.models import resnet50, ResNet50_Weights

class ContrastiveLoss(nn.Module):
    """
    This class will implement the contrastive loss presented in SimClr and follow the implementation in CLIP
    It is important to have this implemented as a module so that we can set an optimizer and compute backprop
    """

    def __init__(self, device, temp=0.01):
        """
        Initialize the loss, uses cross-entropy and needs a tempature to be set.
        """
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.cross_entropy = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, image_encoding, text_encoding):
        """
        implements the contrastive loss function
        """
        # important to determine positive pairs within the batch
        batch_size = image_encoding.shape[0]

        # Ensure features are normalized for proper cosine similarity (uses L2 norm) (set dim 1 to avoid batch)
        image_encoding = F.normalize(image_encoding, p=2, dim=1)
        text_encoding = F.normalize(text_encoding, p=2, dim=1)

        # Calculate cosine similarity between all image-text pairs (since each are normalized, just dot product)
        # Since vectors are normalized, dot product equals cosine similarity
        logits = torch.matmul(image_encoding, text_encoding.T) / self.temp

        # Labels for both directions (image->text and text->image)
        # We want the diagonal elements to match (each image with its corresponding text)
        labels = torch.arange(batch_size, device=self.device)

        # Calculate loss in both directions and average
        image_to_text_loss = self.cross_entropy(logits, labels)
        text_to_image_loss = self.cross_entropy(logits.T, labels)

        # Total loss is the average of both directions
        total_loss = (image_to_text_loss + text_to_image_loss) / 2.0

        return total_loss

class ConstrastiveModel(nn.Module):
    """
    One class that will hold all NN models and layers including BERT, ResNet50, Image projection and text projection
    """

    def __init__(self):
        """
        Creates the torch Module and will initilize all models and projections needed.
        """
        super(ConstrastiveModel, self).__init__()
        # Load pre-trained BERT for text encoding
        self.text_encoder = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")
        # Load pre-trained ResNet for image encoding
        self.image_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Replace final fc layer to match embedding dimension with text encoder
        # I found that this is important to do bc ResNet50 is trained for classification. We don't want this output. Replacing the last layer might help it "forget" this
        embedding_dim = self.text_encoder.config.hidden_size # I found that they have this online
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, embedding_dim) # as for this - checkout the _init_ method in https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L284

        # Projection layers to align embeddings
        self.text_projection = nn.Linear(embedding_dim, embedding_dim)
        self.image_projection = nn.Linear(embedding_dim, embedding_dim)

    def encode_descriptions(self, tokens, attention_mask):
        # get the embedding of the description from BERT
        output = self.text_encoder(input_ids=tokens, 
                                   attention_mask=attention_mask,
                                   return_dict=True)
        # Use [CLS] token embedding as text representation
        text_features = output.last_hidden_state[:, 0, :]
        # now apply the projection layer to the embedding
        text_features = self.text_projection(text_features)

        return text_features

    def encode_image(self, image):
        # Get image embedding after applying resnet
        image = self.image_encoder(image)
        # Now apply projection  layer to the embedding
        image = self.image_projection(image)
        return image
    

    def forward(self, image, tokens, attention_mask):
        # get latent space encodings for image and description
        text_encoding = self.encode_descriptions(tokens, attention_mask)
        image_encoding = self.encode_image(image)
        
        return image_encoding, text_encoding


class Trainer():

    def __init__(self, model, dataloader, device):
         # Pytorch DataLoader
        self.dataloader = dataloader # tokenizer must be set to: AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
        # sets the Contrastive loss function
        self.device = device
        self.contrasitve_loss = ContrastiveLoss(self.device)
        self.model = model.to(self.device)
        # get the layers that will be trained
        # Only train the projection layers and the ResNet fc layer, keep pre-trained weights frozen
        self.params_to_train = list(self.model.text_projection.parameters()) + \
                            list(self.model.image_projection.parameters()) + \
                            list(self.model.image_encoder.fc.parameters())
        # set optimizer
        self.optimizer = optim.AdamW(self.params_to_train, lr=5e-5)
        self.training_number = 1

    def train(self, epochs=5):
        print(f"Starting training run: {self.training_number}")
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (images, tokens, att_mask) in enumerate(self.dataloader):
                # move data to correct device
                images = images.to(self.device)
                tokens = tokens.to(self.device)
                att_mask = att_mask.to(self.device)

                # clear optimizer gradients
                self.optimizer.zero_grad()

                # compute forward pass
                image_embedding, text_embedding = self.model(images, tokens, att_mask)

                # compute loss
                loss = self.contrasitve_loss(image_embedding, text_embedding)

                # compute gradient and update weights
                loss.backward()
                self.optimizer.step()
                # add to loss
                total_loss += loss.item()

                if (batch_idx+1) % 25 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(self.dataloader)}, Loss: {loss.item():.4f}")
            
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(self.dataloader):.4f}")
        self.training_number += 1

    def save_model(self, path="models/clip_model.pth"):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path="models/clip_model.pth"):
        model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    dataloader = None
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using: {device}")
    model = ConstrastiveModel()
    trainer = Trainer(model, dataloader, device)
    trainer.train()
    trainer.save_model()