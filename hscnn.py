import torch
from torch import nn, optim, tensor
import pytorch_lightning as pl
import pandas as pd
import torchmetrics
import json
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
import math
from collections import Counter

from zmq import device

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PairDataset(Dataset):

    def __init__(self, set_type, max_len, batch_size):
        train = open('data/train_pairs.json', 'r')
        train = pd.DataFrame([json.loads(line) for line in train])
        val = open('data/val_pairs.json', 'r')
        val = pd.DataFrame([json.loads(line) for line in val])

        # Load your own tokenizer
        tokenizer = None
        
        if set_type == 'train':
            label = train['label'].tolist()
            toks1 = tokenizer(train['text1'].apply(lambda x : ' '.join(x)).tolist(), padding='max_length', truncation=True, max_length=max_len)['input_ids']
            toks2 = tokenizer(train['text2'].apply(lambda x : ' '.join(x)).tolist(), padding='max_length', truncation=True, max_length=max_len)['input_ids']
            onehot1 = train['onehot1'].tolist()
            onehot2 = train['onehot2'].tolist()
        elif set_type == 'val':
            label = val['label'].tolist()
            toks1 = tokenizer(val['text1'].apply(lambda x : ' '.join(x)).tolist(), padding='max_length', truncation=True, max_length=max_len)['input_ids']
            toks2 = tokenizer(val['text2'].apply(lambda x : ' '.join(x)).tolist(), padding='max_length', truncation=True, max_length=max_len)['input_ids']
            onehot1 = val['onehot1'].tolist()
            onehot2 = val['onehot2'].tolist()
        if len(train) % batch_size:
            
            cut = batch_size * int(len(train) / batch_size)
            #print('CUT ',len(train), batch_size, len(train) % batch_size, cut)
            label = label[:cut]
            toks1 = toks1[:cut]
            toks2 = toks2[:cut]
            onehot1 = onehot1[:cut]
            onehot2 = onehot2[:cut]
        self.label=torch.FloatTensor(label)
        self.toks1=torch.LongTensor(toks1)
        self.toks2=torch.LongTensor(toks2)
        self.onehot1=torch.LongTensor(onehot1)
        self.onehot2=torch.LongTensor(onehot2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        return self.label[idx],self.toks1[idx],self.toks2[idx],self.onehot1[idx],self.onehot2[idx]

class ConvDataset(Dataset):
    
    def __init__(self, set_type = 'train', max_len=1000, batch_size=100):

        # Load your own tokenizer
        tokenizer = None
        
        train = open('data/compare_data_5.json', 'r')
        train = pd.DataFrame([json.loads(line) for line in train])
        test = open('data/compare_data_5.json', 'r')
        test = pd.DataFrame([json.loads(line) for line in test])
        if set_type == 'train':
            x = tokenizer(train['text'].apply(lambda x : ' '.join(x)).tolist(), padding='max_length', truncation=True, max_length=max_len)['input_ids']
            y = train['onehot'].tolist()
        elif set_type == 'test':
            x = tokenizer(test['text'].apply(lambda x : ' '.join(x)).tolist(), padding='max_length', truncation=True, max_length=max_len)['input_ids']
            y = test['onehot'].tolist()
        else:
            print("Building empty attributes as wrong dataset type")
            x = torch.tensor([])
            y = torch.tensor([])
        if len(x) % batch_size:
            x = x[:batch_size * int(len(x) / batch_size)]
            y = y[:batch_size * int(len(x) / batch_size)]
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    
    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)

class HSCNN(pl.LightningModule):
    def __init__(self, vocab_size, embedding_size, output_size, in_channels, out_channels, kernel_size, stride, padding, keep_probab, d, dropout=0.5):
        super(HSCNN,self).__init__()

        self.batch_size = BATCH_SIZE

        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.embedding_size1 = embedding_size
        self.keep_probab = keep_probab
        
        self.d = self.class_num

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.word_embeddings.requires_grad_ = False
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_size[0], embedding_size), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_size[1], embedding_size), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_size[2], embedding_size), stride, padding)
        self.dropout = nn.Dropout(keep_probab)
        self.liner = nn.Linear(len(kernel_size)*out_channels, 1024)
        self.label = nn.Linear(1024, self.d)
        self.sig = nn.Sigmoid()
        self.out = nn.Linear(1024, output_size)
        self.liner_onehot = nn.Linear(self.d, 1024)
        self.rel = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.lr = 1e-5
        self.weight_decay = 1e-6

        # Compute weights for imbalanced labels
        alpha_dataset = self.train_dataloader() # Training dataset for alpha computation
        train_labels = [elem.item() for sim, x1, x2, y1, y2 in alpha_dataset.dataset for f in (lambda x:torch.nonzero(x[0]).squeeze(1), lambda x:torch.nonzero(x[1]).squeeze(1)) for elem in f([y1, y2])]
        c = Counter(train_labels)
        N = 2 * len(alpha_dataset) # The dataset comes in pairs
        Nc = [c[x] if x in c.keys() else 0.0 for x in range(207)]
        alpha = torch.tensor([math.log(N/Nc[x]) for x in range(207)])
        self.loss = nn.BCEWithLogitsLoss(weight=alpha)
        self.sloss = nn.BCEWithLogitsLoss()

        
        self.f1 = torchmetrics.F1Score(num_classes=self.class_num, multiclass=False, average='micro')
        self.f1_all = torchmetrics.F1Score(num_classes=self.class_num, multiclass=False, average=None)
        self.f1_siam = f1 = torchmetrics.F1Score(num_classes=1, multiclass=False, average='micro')
        self.recall = torchmetrics.Recall(num_classes=self.class_num, threshold=0.5, average=None, top_k=3)

        #Build centroids
        emb = torch.tensor([])
        data = self.prototype_dataloader()
        self.to(device)
        for x,y in data:
            tmp = x.to(device)
            tmp = self.get_representation(tmp)
            emb = torch.cat((emb,tmp.detach().cpu()), dim=0)
        class_tensors = data.dataset.y
        prototype = torch.mm(emb.T, class_tensors.float())
        prototype = prototype / torch.sum(class_tensors,dim=0)
        self.prototype = prototype.T

    def prototype_dataloader(self):
        return DataLoader(ConvDataset(set_type='train', max_len=512, batch_size=3), batch_size=3, shuffle=True, num_workers=16)  

    def train_dataloader(self):
        train_data = PairDataset(set_type='train', max_len=512, batch_size=self.batch_size)
        return DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=16)
        
    def val_dataloader(self):
        val_data = PairDataset(set_type='val', max_len=512, batch_size=self.batch_size)
        return DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=16)
    
    def test_dataloader(self):
        test_data = ConvDataset(set_type='test', max_len=512, batch_size=self.batch_size)
        return DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=16)
    
    def conv_pool(self, input_t, conv_layer):
        conv_out = conv_layer(input_t)
        activation = F.relu(conv_out.squeeze(3))
        pool_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return pool_out

    def get_representation(self, input_t):
        input_t = self.word_embeddings(input_t)
        input_t = input_t.unsqueeze(1)
        pool_out1 = self.conv_pool(input_t, self.conv1)
        pool_out2 = self.conv_pool(input_t, self.conv2)
        pool_out3 = self.conv_pool(input_t, self.conv3)
        all_pool_out = torch.cat((pool_out1, pool_out2, pool_out3), 1)
        representation = self.liner(all_pool_out)
        return representation
    
    def forward_one(self, input_sentences, onehot):

        input_t = self.word_embeddings(input_sentences)
        self.dropout1 = nn.Dropout(0.25)
        input_t = self.dropout1(input_t)
        input_t = input_t.unsqueeze(1)

        pool_out1 = self.conv_pool(input_t, self.conv1)
        pool_out2 = self.conv_pool(input_t, self.conv2)
        pool_out3 = self.conv_pool(input_t, self.conv3)
        all_pool_out = torch.cat((pool_out1, pool_out2, pool_out3), 1)
        x = self.dropout(all_pool_out)
        
        x = self.liner(x)
        cnn_out = self.label(x)
        q_w = self.liner_onehot(onehot.float())
        q_w = self.rel(q_w / math.sqrt(self.d))
        return x, q_w, cnn_out
    

    def forward(self, toks1, toks2, onehot1, onehot2):
        x1, q_w1, cnn_out1 = self.forward_one(toks1, onehot1)
        out1 = self.sig(x1)
        x2, q_w2, cnn_out2 = self.forward_one(toks2, onehot2)
        out2 = self.sig(x2)
        dis = torch.abs(x1 - x2)
        tmp = torch.mul(dis, q_w2)
        out = self.out(tmp)
        out = torch.squeeze(out)

        return out1, out2, cnn_out1, cnn_out2, out

    def compute_similarity(self, article_representation, onehot, class_representation):
        q_w = self.liner_onehot(onehot.float())
        q_w = self.rel(q_w / math.sqrt(self.d))
        similarity = torch.abs(article_representation - class_representation.to(self.device))
        similarity = torch.mul(similarity, q_w)
        return torch.squeeze(self.out(similarity))

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        label, toks1, toks2, onehot1, onehot2 = train_batch

        _, _, cnn_out1, cnn_out2, out = self.forward(toks1, toks2, onehot1, onehot2) #(B, C)
        #targets: (B,)

        sim = torch.sum(torch.sigmoid(out)*label)/len(torch.nonzero(torch.sigmoid(out)*label))
        dissim = torch.sum(torch.sigmoid(out)*(1-label))/len(torch.nonzero(torch.sigmoid(out)*(1-label)))
        self.log('train_sim', sim)
        self.log('train_dissim', dissim)

        siamese_loss = self.sloss(out, label.float())
        cnnloss1 = self.loss(cnn_out1, onehot1.float())
        cnnloss2 = self.loss(cnn_out2, onehot2.float())
        loss = siamese_loss + 0.3*cnnloss1 + 0.3*cnnloss2

        self.log('training_loss_step', loss)

        return loss
   

    def validation_step(self, valid_batch, batch_idx):

        label, toks1, toks2, onehot1, onehot2 = valid_batch

        _, _, cnn_out1, cnn_out2, out = self.forward(toks1, toks2, onehot1, onehot2) #(B, C)

        sim = torch.sum(torch.sigmoid(out)*label)/len(torch.nonzero(torch.sigmoid(out)*label))
        dissim = torch.sum(torch.sigmoid(out)*(1-label))/len(torch.nonzero(torch.sigmoid(out)*(1-label)))
        self.log('val_sim', sim)
        self.log('val_dissim', dissim)

        siamese_loss = self.sloss(out.unsqueeze(1), label.unsqueeze(1).float())
        cnnloss1 = self.loss(cnn_out1, onehot1.float())
        cnnloss2 = self.loss(cnn_out2, onehot2.float())
        loss = siamese_loss + 0.3*cnnloss1 + 0.3*cnnloss2

        probs1 = torch.sigmoid(cnn_out1)
        probs2 = torch.sigmoid(cnn_out2)
        probs = torch.sigmoid(out)

        predictions1 = torch.gt(probs1.detach(), 0.5).long()
        predictions2 = torch.gt(probs2.detach(), 0.5).long()
        predictions = torch.gt(probs.detach(), 0.5).long()

        f1_avg = (self.f1(predictions1, onehot1.long()) + self.f1(predictions2, onehot2.long())) / 2

        f1_sim = self.f1_siam(predictions, label.long())

        self.log('validation_f1_step', f1_avg, prog_bar=True)
        self.log('validation_f1_sim', f1_sim)
        
        self.log('validation_loss_step', loss, prog_bar=True)


    
    def test_step(self, test_batch, batch_idx):
        toks, onehot = test_batch
        batch_size = toks.shape[0]
        probs = torch.zeros((batch_size,self.class_num)).float()
        for class_id in range(self.class_num):
            probs[:, class_id] = self.compute_similarity(self.get_representation(toks), torch.nn.functional.one_hot(torch.tensor([class_id]),num_classes=self.class_num).repeat((batch_size,1)).to(self.device), self.prototype[class_id])
        predictions = torch.gt(probs.detach(), 0.5).long()
        return probs, predictions, onehot
    
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
