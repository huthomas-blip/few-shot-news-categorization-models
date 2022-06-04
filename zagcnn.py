
import torch
from torch import nn, optim, tensor
import pytorch_lightning as pl
import pandas as pd
import torchmetrics
from torch_geometric.data import Data
import torch_geometric
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

BATCH_SIZE = 64

class NewsDataset(Dataset):

  def __init__(self, set_type = 'train', max_len=1000):
    batch_size = BATCH_SIZE
    # Add you code to load the data here (tokens and labels) for train, valid and test splits, as well as token to idx and idx to token mappings and label names
    train = pd.DataFrame() # load train data
    val = pd.DataFrame() # load valid data
    test = pd.DataFrame() # load test data
    ixtoword = {} # token to idx mapping
    wordtoix = {} # idx to token mapping
    class_names = [] # class names

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
    self.class_names = class_names

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

# model
class ZAGCNN(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=100, ngram=55, dropout=0.5, class_names=None, class_id=None):
        super(ZAGCNN,self).__init__()
        
        self.class_num = len(class_names)
        self.class_name = class_names
        self.sentence_len = 512


        self.batch_size = BATCH_SIZE
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # Load your own embeddings here
        self.embedding.requires_grad_ = False

        #compute class embeddings
        self.embedding_class = nn.Embedding(self.class_num, embedding_dim) #Load you own class embeddings here
        self.embedding_class.requires_grad_ = False

        #Build label graph
        label_dict = {}
        for idx, l in enumerate(class_id):
            label_dict[l] = {'id':idx, 'embedding':self.embedding_class[idx,:], 'children':[]}
        for l in label_dict.keys():
            for l2 in label_dict.keys():
                if l==l2[:-3]:
                    label_dict[l]['children'].append(label_dict[l2]['id'])

        x = self.embedding_class
        edge1, edge2 = [], []
        for l in label_dict.keys():
            edge1 += len(label_dict[l]['children']) * [label_dict[l]['id']]
            edge2 += label_dict[l]['children']
            edge2 += len(label_dict[l]['children']) * [label_dict[l]['id']]
            edge1 += label_dict[l]['children']
        edge_index = torch.tensor([edge1, edge2], dtype=torch.long)
        self.graph = Data(x=x, edge_index=edge_index)

        u = 256
        self.conv = torch.nn.Conv1d(embedding_dim, u, 2*ngram+1)
        q=512
        self.conv1 = torch_geometric.nn.GCNConv(embedding_dim, q)
        self.conv2 = torch_geometric.nn.GCNConv(q, q)
        
        self.hidden = nn.Linear(embedding_dim, embedding_dim)
        self.layer = nn.Linear(u, embedding_dim)
        self.layer2 = nn.Linear(u, q+embedding_dim)
        self.out = nn.Linear(self.class_num, self.class_num)
        self.tanh = torch.tanh
        self.dropout = nn.Dropout(dropout)

        self.lr = 4e-7
        self.weight_decay = 1e-8
        self.loss = nn.BCEWithLogitsLoss()

        self.f1 = torchmetrics.F1Score(num_classes=self.class_num, multiclass=False, average='micro')
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=self.class_num, multilabel=True)

    def train_dataloader(self):
        train_data = NewsDataset(set_type='train', max_len=self.sentence_len)
        return DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=16)
        
    def val_dataloader(self):
        val_data = NewsDataset(set_type='val', max_len=self.sentence_len)
        return DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=16)
    
    def test_dataloader(self):
        test_data = NewsDataset(set_type='test', max_len=self.sentence_len)
        return DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=16)

        
    def forward(self,inputs):
        emb = self.embedding(inputs) # (B, L, e)
        emb = torch.transpose(emb, 1, 2)

        d = self.dropout(emb)
        d = self.conv(d)
        d = torch.transpose(d, 1, 2)

        class_tensor = torch.tensor([[i for i in range(self.class_num)] for j in range(inputs.size(0))], device=self.device)

        emb_c = self.embedding_class(class_tensor)
        v=emb_c
        d2 = self.dropout(d)
        d2 = self.layer(d2)
        d2 = self.tanh(d2)
        
        a = F.softmax(torch.bmm(d2,torch.transpose(v,1,2)), dim=2)
        #print("a: ", a.shape)

        c = torch.bmm(torch.transpose(a, 1, 2),d)
        x, edge_index = self.graph['x'].to(self.device), self.graph['edge_index'].to(self.device)
        v1 = self.conv1(x, edge_index)
        v2 = self.conv2(v1, edge_index)
        self.graph.detach()

        v2_batched = v2.repeat((self.batch_size,1,1))

        v3 = torch.cat((v,v2_batched), dim=2)

        e = self.dropout(c)
        e = self.layer2(e)
        e = F.relu(e)

        y = torch.sum(torch.bmm(e, torch.transpose(v3, 1, 2)), dim=2)
        y = self.dropout(y)
        y = self.out(y)
        
        return y

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
        #self.confusion_matrix(probs, targets.long())

        self.log('validation_f1_step', self.f1, prog_bar=True)
        self.log('validation_loss_step', loss, prog_bar=True)
        return (predictions, targets)

    def test_step(self, valid_batch, batch_idx):

        inputs,targets = valid_batch

        preds = self.forward(inputs.view(self.batch_size,-1)).float() #(B, C)
        
        loss = self.loss(preds, targets.float())

        probs = torch.sigmoid(preds)

        predictions = torch.gt(probs.detach(), 0.5).long()

        self.f1(predictions, targets.long())
        cm = self.confusion_matrix(predictions, targets.long())
        print(cm)

        self.log('test_f1_step', self.f1, prog_bar=True)
    