import torch
from torch import nn, optim, tensor
import pytorch_lightning as pl
import pandas as pd
import torchmetrics
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

BATCH_SIZE = 64

class NewsDataset(Dataset):

    def __init__(self, set_type = 'train', max_len=1000):
        
        # Add you code to load the data here (tokens and labels) for train, valid and test splits, as well as token to idx and idx to token mappings and label names
        train = pd.DataFrame() # load train data
        val = pd.DataFrame() # load valid data
        test = pd.DataFrame() # load test data
        ixtoword = {} # token to idx mapping
        wordtoix = {} # idx to token mapping
        class_names = [] # class names
        batch_size = BATCH_SIZE

        if set_type == 'train':
            x = train['tokens'].tolist()
            y = train['labels'].tolist()
        elif type == 'val':
            x = val['tokens'].tolist()
            y = val['labels'].tolist()
        else:
            x = test['tokens'].tolist()
            y = test['labels'].tolist()
        if len(x) % batch_size:
            x = x[:batch_size * int(len(x) / batch_size)]
            y = y[:batch_size * int(len(x) / batch_size)]
        self.x_train=torch.LongTensor(x)
        self.y_train=torch.LongTensor(y)
        self.ixtoword = ixtoword
        self.wordtoix = wordtoix
        self.class_name = class_names

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]


# model
class LEAM(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=100, ngram=55, dropout=0.5):
        super(LEAM,self).__init__()
        
        self.batch_size = BATCH_SIZE
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # You can load your own word embeddings here
        self.embedding.requires_grad_ = False

        self.embedding_class = nn.Embedding(self.class_num, embedding_dim) # You can load your own class embeddings here
        self.embedding_class.requires_grad_ = False

        self.conv = torch.nn.Conv1d(self.class_num, self.class_num, 2*ngram+1,padding=ngram)
        
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.layer = nn.Linear(hidden_dim, self.class_num)

        self.dropout = nn.Dropout(dropout)

        #Learning parameters
        self.lr = 1e-3
        self.weight_decay = 1e-6
        self.loss = nn.BCEWithLogitsLoss()

        # Evalutation metrics
        self.f1 = torchmetrics.F1Score(num_classes=self.class_num, multiclass=False, average='micro')
        self.f1_all = torchmetrics.F1Score(num_classes=self.class_num, multiclass=False, average=None)
        self.recall = torchmetrics.Recall(num_classes=self.class_num, threshold=0.5, average=None, top_k=3)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=self.class_num, multilabel=True)

        self.results = {'f1': {}, 'r3': {}, 'f1_micro': {}}

    def train_dataloader(self):
        train_data = NewsDataset(set_type='train', max_len=1000)
        return DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=16)
        
    def val_dataloader(self):
        val_data = NewsDataset(set_type='val', max_len=1000)
        return DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=16)
    
    def test_dataloader(self):
        test_data = NewsDataset(set_type='test', max_len=1000)
        return DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=16)

    def forward(self,inputs):
        
        emb = self.embedding(inputs) # (B, L, e)
        
        embn = torch.norm(emb, p=2, dim=2).detach()        
        emb_norm = emb.div(embn.unsqueeze(2))

        class_tensor = torch.tensor([[i for i in range(self.class_num)] for j in range(inputs.size(0))], device=self.device)

        emb_c = self.embedding_class(class_tensor)

        emb_cn = torch.norm(emb_c, p=2, dim=2).detach()
        emb_c_norm = emb_c.div(emb_cn.unsqueeze(2))
        
        emb_norm_t = emb_norm.permute(0, 2, 1) # (B, e, L)
        
        g = torch.bmm(emb_c_norm,emb_norm_t) #(B, C, L)
        
        g = self.dropout(g)

        g = F.relu(self.conv(g))
        

        beta = torch.max(g,1)[0].unsqueeze(2) #(B, L)
        
        beta = F.softmax(beta,1) #(B, L)
        
        z = torch.mul(beta,emb) #(B, L, e)
        
        z = z.sum(1) #(B, e)

        z = self.dropout(z)
        z = self.hidden(z)
        
        
        z = self.dropout(z)
        out = self.layer(z) #(B, C)
        
        logits = out

        return logits

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        inputs,targets = train_batch

        preds = self.forward(inputs.view(self.batch_size,-1)).float() #(B, C)
        #targets: (B,)
    
        loss = self.loss(preds, targets.float())

        probs = torch.sigmoid(preds)

        predictions = torch.gt(probs.detach(), 0.5).long()

        self.log('training_loss_step', loss)

        return loss
   

    def validation_step(self, valid_batch, batch_idx):

        inputs,targets = valid_batch

        preds = self.forward(inputs.view(self.batch_size,-1)).float() #(B, C)
        
        loss = self.loss(preds, targets.float())

        probs = torch.sigmoid(preds)

        predictions = torch.gt(probs.detach(), 0.5).long()

        self.f1(predictions, targets.long())

        self.log('validation_f1_step', self.f1, prog_bar=True)
        self.log('validation_loss_step', loss, prog_bar=True)
        return (predictions, targets)


    def test_step(self, valid_batch, batch_idx):

        inputs,targets = valid_batch

        preds = self.forward(inputs.view(self.batch_size,-1)).float() #(B, C)
        
        loss = self.loss(preds, targets.float())

        probs = torch.sigmoid(preds)

        predictions = torch.gt(probs.detach(), 0.5).long()


        return probs, predictions, targets

    def test_epoch_end(self, test_step_outputs):
        probs, predictions, targets = [x[0] for x in test_step_outputs], [x[1] for x in test_step_outputs], [x[2] for x in test_step_outputs]
        probs = torch.cat(probs, 0)
        predictions = torch.cat(predictions, 0)
        targets = torch.cat(targets, 0)
        probs = probs.detach().cpu()
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()
        self.f1_all.to('cpu')
        self.f1.to('cpu')
        self.recall.to('cpu')
        test_f1_micro = self.f1(predictions, targets.long())
        test_f1 = self.f1_all(predictions, targets.long())
        test_recall = self.recall(probs, targets.long())

        metrics = {'f1_micro': test_f1_micro, 'f1': test_f1, 'r3': test_recall}
        
        self.results = metrics
        return metrics    