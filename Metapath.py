import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm import tqdm

from reading_data import DataReader, Metapath2vecDataset
from model import SkipGramModel
from download import CustomDataset
import numpy as np
np.random.seed(80000)
import pdb
from tensorboardX import SummaryWriter
class Metapath2VecTrainer:
    def __init__(self, path ,output_file, dim, window_size, iterations, batch_size, care_type, initial_lr, min_count, num_workers):

        dataset = CustomDataset(path)

        self.data = DataReader(dataset, min_count, care_type)
        dataset = Metapath2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers, collate_fn=dataset.collate)

        self.output_file_name = output_file
        # if self.data.hasnone == True:
        #     self.emb_size = len(self.data.word2id) - 1  
        # else:
        #     self.emb_size = len(self.data.word2id)
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = dim
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        # self.device = "cpu"
        self.device = torch.device("cuda:1" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.to(self.device)
            # self.skip_gram_model.cuda()

    def train(self):
        
        torch.autograd.set_detect_anomaly(True)
        optimizer = optim.SparseAdam(list(self.skip_gram_model.parameters()), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))  # 余弦退火学习率更新策略
        writer = SummaryWriter('runs/exp1')
        loss_fn = nn.CrossEntropyLoss()
        
        net = nn.Sequential(nn.Linear(self.emb_dimension*2, 1))    
        if self.use_cuda:           
            net.to(self.device)
            # net0.to(self.device)
        init.normal_(net[0].weight, mean=0, std=1)
        init.constant_(net[0].bias, val=0) 
        
        optimizers = torch.optim.RMSprop(net.parameters(), lr=0.001)
        x = 0
        for iteration in range(self.iterations):
            print("\n\n\nIteration: " + str(iteration + 1))
            running_loss = 0.0

            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)
                    # print("pos_u:",len(pos_u),pos_u)
                    # print("pos_v:",len(pos_v),pos_v)
                    # print("neg_v:",len(neg_v),neg_v)
                    
                    # pdb.set_trace()
                    pos1 ,pos2, neg1, neg2 = self.skip_gram_model(pos_u, pos_v, neg_v, self.emb_size)
                    
                    # target1 = torch.ones(len(pos1)).numpy()
                    # target1 = torch.LongTensor(target1).to(self.device)
                    # target0 = torch.zeros(len(neg1)).numpy()
                    # target0 = torch.LongTensor(target0).to(self.device)
                    # pos1 = net(pos1)
                    # neg1 = net(neg1)
                    # loss1 = loss_fn(pos1,target1) + loss_fn(neg1,target0)
                    data1 = net(pos1)
                    data1 = -F.logsigmoid(data1)
                    
                    data3 = net(neg1)
                    data3 = -F.logsigmoid(-data3)
                   
                    loss1 = torch.mean(data1 + data3)
                    optimizers.zero_grad()
                    loss1.backward(retain_graph=True) 
                    optimizer.zero_grad()
                  
                    loss = loss1
                    loss.backward()
                    optimizers.step()
                    optimizer.step()
                    scheduler.step()
                    running_loss = running_loss * 0.9 + loss.item() * 0.1

                    if i > 0 and i % 200 == 0:
                        x = x + 1
                        # writer.add_scalar('loss',running_loss,x)
                        print(" Loss: " + str(running_loss))

        self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


