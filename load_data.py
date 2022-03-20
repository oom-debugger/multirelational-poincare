class Data:

    def __init__(self, data_dir="data/WN18RR/"):
                # if data/drkg/drkg.tsv
        if data_dir.lower()=='data/drkg/':
            self.maybe_load_drkg()
            if not (os.path.isfile('data/drkg/train.txt') and
                    os.path.isfile('data/drkg/valid.txt') and
                    os.path.isfile('data/drkg/test.txt')):
                self.make_drkg_data_splits(drkg_file='data/drkg/drkg.tsv')

        self.train_data = self.load_data(data_dir, "train")
        self.valid_data = self.load_data(data_dir, "valid")
        self.test_data = self.load_data(data_dir, "test")
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train"):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding="utf-8") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def maybe_load_drkg(self):
        drkg_file = 'data/drkg/drkg.tsv'
        download_and_extract(drkg_file)

    def make_drkg_data_splits(self, drkg_file):
        if not os.path.isfile(drkg_file):
            raise ValueError('drkg file does not exist...')
        df = pd.read_csv(drkg_file, sep="\t")
        triples = df.values.tolist()

        num_triples = len(triples)
        num_triples
        # Please make sure the output directory exist.
        seed = np.arange(num_triples)
        np.random.shuffle(seed)
        
        train_cnt = int(num_triples * 0.9)
        valid_cnt = int(num_triples * 0.05)
        train_set = seed[:train_cnt]
        train_set = train_set.tolist()
        valid_set = seed[train_cnt:train_cnt+valid_cnt].tolist()
        test_set = seed[train_cnt+valid_cnt:].tolist()
        
        with open("data/drkg/train.txt", 'w+') as f:
            for idx in train_set:
                f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))                
        with open("data/drkg/valid.txt", 'w+') as f:
            for idx in valid_set:
                f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))        
        with open("data/drkg/test.txt", 'w+') as f:
            for idx in test_set:
                f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

########################### DrKG ###################################
# For DRKG
import os
import tarfile
import shutil
import requests
import pandas as pd
import numpy as np
import dgl
import sys
from pathlib import Path


def download_and_extract(drkg_file, path=None, filename=None):
    if os.path.exists(drkg_file):
        return drkg_file

    if not path:
        path = "/tmp/data/"
    if not filename:
        filename = "drkg.tar.gz"

    fn = os.path.join(path, filename)
    if not os.path.isfile(fn):
        print('%s does not exist..downloading..' % fn)
        url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/DRKG/drkg.tar.gz"
        f_remote = requests.get(url, stream=True)
        sz = f_remote.headers.get('content-length')
        assert f_remote.status_code == 200, 'fail to open {}'.format(url)
        with open(filename, 'wb') as writer:
            for chunk in f_remote.iter_content(chunk_size=1024*1024):
                writer.write(chunk)
        print('Download finished. Unzipping the file...')
    else:
        print('tar file already exists! Unzipping...')
        
    parent_dir = Path(drkg_file).parent
    if not (os.path.exists(parent_dir) and os.path.isdir(parent_dir)):
        os.makedirs(path, exist_ok=True)
    file = tarfile.open(fn, mode='r:gz')
    file.extractall(path=str(parent_dir))
    file.close()



def insert_entry(entry, ent_type, dic):
    if ent_type not in dic:
        dic[ent_type] = {}
    ent_n_id = len(dic[ent_type])
    if entry not in dic[ent_type]:
         dic[ent_type][entry] = ent_n_id
    return dic
    
    
def create_entity_dictionary_with_dgl(drkg_file, path=None, filename=None):
#    drkg_file = '/home/mehrdad/github_dir/multirelational-poincare/data/drkg/drkg.tsv'
    download_and_extract(drkg_file, path, filename)
    print ('drkg_file:', drkg_file)

    df = pd.read_csv(drkg_file, sep ="\t", header=None)
    triplets = df.values.tolist()
    print ('Extracting Triplets......')
    entity_dictionary = {}    
    for triple in triplets:
        src = triple[0]
        split_src = src.split('::')
        src_type = split_src[0]
        dest = triple[2]
        split_dest = dest.split('::')
        dest_type = split_dest[0]
        insert_entry(src,src_type,entity_dictionary)
        insert_entry(dest,dest_type,entity_dictionary)
    # Create a dictionary of relations: the key is the relation and the 
    # value is the list of (source node ID, destimation node ID) tuples.
    
    print ('Extracting Edge Dictionary......')
    edge_dictionary={}
    for triple in triplets:
        src = triple[0]
        split_src = src.split('::')
        src_type = split_src[0]
        dest = triple[2]
        split_dest = dest.split('::')
        dest_type = split_dest[0]
        
        src_int_id = entity_dictionary[src_type][src]
        dest_int_id = entity_dictionary[dest_type][dest]
        
        pair = (src_int_id,dest_int_id)
        etype = (src_type,triple[1],dest_type)
        if etype in edge_dictionary:
            edge_dictionary[etype] += [pair]
        else:
            edge_dictionary[etype] = [pair]
            
    print ('Creating Hetrograph....')
    graph = dgl.heterograph(edge_dictionary);
    total_nodes = 0;
    for ntype in graph.ntypes:
        print(ntype, '\t', graph.number_of_nodes(ntype));
        total_nodes += graph.number_of_nodes(ntype);
    print("Graph contains {} nodes from {} node-types.".format(total_nodes, len(graph.ntypes)))
    # Graph(num_nodes: Dict[str, int], num_edges: Dict[Tuple[str, str, str], int], metagraph: List[Tiple[str, str]])
    return graph
