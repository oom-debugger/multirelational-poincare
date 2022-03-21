import numpy as np
import torch
import time
from collections import defaultdict
from load_data import Data
from model import MuRP, MuRE
from rsgd import RiemannianSGD
import argparse

import os

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def get_er_vocab(data, idxs=[0, 1, 2]):
    er_vocab = defaultdict(list)
    for triple in data:
        er_vocab[(triple[idxs[0]], triple[idxs[1]])].append(triple[idxs[2]])
    return er_vocab


def setup_distributed_env(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

class DataLoaer:
    
    def __init__(self, data_dir, rank, world_size, batch_size, pin_memory=False, num_workers=0, distributed=False):
        self.dataset = Data(data_dir=data_dir)
        # TODO(khatir): shuffle data here...
        self._entity_idxs = {self.dataset.entities[i]:i for i in range(len(self.dataset.entities))}
        self.relation_idxs = {self.dataset.relations[i]:i for i in range(len(self.dataset.relations))}
        self.sr_vocab = get_er_vocab(self.get_data_idxs(self.dataset.data))
        self.er_vocab = get_er_vocab(self.get_data_idxs(self.dataset.train_data))
        
        # Do NOT use train_data_idxs inside the training loop as it does not
        # provide correct indexed for sampled data for each worker.//
        train_data_idxs = self.get_data_idxs(self.dataset.train_data)
        print("Number of training data points for all batches: %d" % len(train_data_idxs))

        dataset = Data(data_dir=data_dir)
        if distributed:
            sampler = DistributedSampler(dataset.train_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        else:
            sampler = None
        self.train_dataloader = DataLoader(
                dataset.train_data, batch_size=batch_size, 
                pin_memory=pin_memory, num_workers=num_workers, 
                drop_last=False, shuffle=False, sampler=sampler)

    @property
    def entities(self):
        return self.dataset.entities
    
    @property
    def relations(self):
        return self.dataset.relations

    @property
    def entity_idxs(self):
        return self._entity_idxs

    @property
    def test_data(self):
        return self.dataset.test_data
 
    def get_data_idxs(self, data):
        data_idxs = [(self._entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self._entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_test_data_idxs(self):
        return self.get_data_idxs(self, self.dataset.test_data)

    
class Experiment:

    def __init__(self, data_dir, learning_rate=50, dim=40, nneg=50, model="poincare",
                 num_iterations=500, batch_size=128, cuda=False, 
                 rank=0, world_size=1):
        self.model = model
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.cuda = cuda
        self.distributed = world_size > 1
        self.rank = rank
        if self.distributed:  # i.e. distributed:
            # setup the process groups
            setup_distributed_env(rank, world_size)
        # prepare the dataloader, note that it must be called after setup
        self.data_loader = DataLoaer(data_dir, rank, world_size, batch_size)
        self.train_dataloader = self.data_loader.train_dataloader
       
    def __del__ (self):
        if self.distributed:
            dist.destroy_process_group()


    def evaluate(self, model):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])
                
        test_data_idxs = self.data_loader.get_test_data_idxs()
        print("Number of data points: %d" % len(test_data_idxs))
        for i, data_point in enumerate(test_data_idxs):
            e1_idx = torch.tensor(data_point[0])
            r_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions_s = model.forward(
                    e1_idx.repeat(len(self.data_loader.entities)), 
                    r_idx.repeat(len(self.data_loader.entities)), 
                    range(len(self.dataset.entities)))

            filt = self.data_loader.sr_vocab[(data_point[0], data_point[1])]
            target_value = predictions_s[e2_idx].item()
            predictions_s[filt] = -np.Inf
            predictions_s[e1_idx] = -np.Inf
            predictions_s[e2_idx] = target_value

            sort_values, sort_idxs = torch.sort(predictions_s, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            rank = np.where(sort_idxs==e2_idx.item())[0][0]
            ranks.append(rank+1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
            
        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))
    

    def train_and_eval(self):
        print("Training the %s model..." %self.model)

        if self.model == "poincare":
            model = MuRP(self.data_loader.entities, self.data_loader.relations, self.dim)
        else:
            model = MuRE(self.data_loader.entities, self.data_loader.relations, self.dim)
        if self.distributed:
            # 1. instantiate the model(it's your own model) and move it to the right device using model.to(rank)
            # 2. wrap the model with DDP
            # device_ids tell DDP where is your model
            # output_device tells DDP where to output, in our case, it is rank
            # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
            model = DDP(model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)
            
        param_names = [name for name, param in model.named_parameters()]
        opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        
        if self.cuda:
            model.cuda()
            
        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()

            losses = []
            for j, data_batch in enumerate(self.train_dataloader):                
                negsamples = np.random.choice(
                        list(self.data_loader.entity_idxs.values()), 
                        size=(data_batch.shape[0], self.nneg))
                
                e1_idx = torch.tensor(np.tile(np.array([data_batch[:, 0]]).T, (1, negsamples.shape[1]+1)))
                r_idx = torch.tensor(np.tile(np.array([data_batch[:, 1]]).T, (1, negsamples.shape[1]+1)))
                e2_idx = torch.tensor(np.concatenate((np.array([data_batch[:, 2]]).T, negsamples), axis=1))

                targets = np.zeros(e1_idx.shape)
                targets[:, 0] = 1
                targets = torch.DoubleTensor(targets)

                opt.zero_grad()
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2_idx = e2_idx.cuda()
                    targets = targets.cuda()

                predictions = model.forward(e1_idx, r_idx, e2_idx)      
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(it)
            print(time.time()-start_train)    
            print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                if not it%5:
                    print("Test:")
                    self.evaluate(model)



def main(rank, 
         world_size,
         data_dir,
         learning_rate,
         batch_size,
         num_iterations, 
         dim, 
         cuda,
         nneg, 
         model):
    # read data
    experiment = Experiment(data_dir=data_dir,
                            learning_rate=learning_rate, batch_size=batch_size, 
                            num_iterations=num_iterations, dim=dim, 
                            cuda=cuda, nneg=nneg, model=model,
                            rank=rank, world_size=world_size)
    experiment.train_and_eval() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WN18RR", nargs="?",
                    help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--model", type=str, default="poincare", nargs="?",
                    help="Which model to use: poincare or euclidean.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--nneg", type=int, default=50, nargs="?",
                    help="Number of negative samples.")
    parser.add_argument("--lr", type=float, default=50, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dim", type=int, default=40, nargs="?",
                    help="Embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--workers", type=int, default=1, nargs="?",
                    help="number of worker to run the training.")

    args = parser.parse_args()
    world_size = args.workers

    torch.backends.cudnn.deterministic = True 
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 

    data_dir = "data/%s/" % args.dataset
    learning_rate=args.lr
    batch_size=args.batch_size
    num_iterations=args.num_iterations
    dim=args.dim
    cuda=args.cuda
    nneg=args.nneg
    model=args.model
    world_size=world_size
  
    mp.spawn(
        main,
        args=(world_size, data_dir, learning_rate, batch_size, num_iterations, 
              dim, cuda, nneg, model),
        nprocs=world_size
    )                

