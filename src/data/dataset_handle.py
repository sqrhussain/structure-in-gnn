import wget
import tarfile
import os
import networkx as nx
import scipy
import shutil
import zipfile
import pandas as pd
import csv
import numpy as np
import json


def create_citation_dataset(dataset_name, url, target_tmp_file, dir_path, target_processed_path,
                            threshold=50, raw_folder_name = None, select_largest_component = True, sample_features = None):  # to use for CORA and CITESEER

    if raw_folder_name is None:
        raw_folder_name = dataset_name

    # download dataset
    if (url is not None) and (not os.path.exists(dir_path + '/' + dataset_name)):
        print(f'Downloading {url}...')
        wget.download(url, target_tmp_file)
        tar = tarfile.open(target_tmp_file, "r:gz")
        tar.extractall(dir_path)
        tar.close()
        os.remove(target_tmp_file)
    print('File downloaded and extracted!')

    # load features
    features_path = f'{dir_path}/{raw_folder_name}/{dataset_name}.content'
    edges_path = f'{dir_path}/{raw_folder_name}/{dataset_name}.cites'
    target = {}
    features = {}
    class_count = {}
    nodes = set()
    with open(features_path, encoding='utf8') as f:
        for line in f:
            if dataset_name == 'twitter':
                info = line.split('\t')
            else:
                info = line.split()
            if len(info) == 1:
                info = line.split(',')
            nodes.add(info[0])
            y = info[-1]
            target[info[0]] = y
            fs = info[1:-1]
            # features[info[0]] = info[1:-1]
            if sample_features is not None:
                np.random.seed(0)
                fs = np.random.choice(np.array(fs), sample_features, replace=False)
                fs = fs.tolist()
            features[info[0]] = fs
            if y in class_count:
                class_count[y] += 1
            else:
                class_count[y] = 1
    edges = []
    with open(edges_path) as f:
        for line in f:
            info = line.split()
            if len(info) == 1:
                info = line.split(',')
            # remove edges connecting (to) nodes with rare labels
            if info[0] in nodes and info[1] in nodes and info[0] != info[1]:
                # convert to suitable representation (citing -> cited)
                edges.append([info[1], info[0]])
    # print(edges_path)
    # select biggest connected component
    undirected_graph = nx.Graph()  # undirected, just to find the biggest connected component
    undirected_graph.add_edges_from(edges)
    # remove nodes with rare labels
    nodes = {u for u in undirected_graph.nodes() if class_count[target[u]] >= threshold}
    if select_largest_component:
        largest_component = max(nx.connected_components(undirected_graph), key=len)
        nodes = largest_component
    if not os.path.exists(f'{target_processed_path}/{dataset_name}'):
        os.mkdir(f'{target_processed_path}/{dataset_name}')
    edges = [e for e in edges if e[0] in nodes and e[1] in nodes]
    with open(f'{target_processed_path}/{dataset_name}/{dataset_name}.cites', 'w') as f:
        for edge in edges:
            f.write(' '.join(edge) + '\n')

    with open(f'{target_processed_path}/{dataset_name}/{dataset_name}.content', 'w', encoding='utf8') as f:
        for node in nodes:
            delimiter = ' '
            f.write(node + delimiter)
            f.write(delimiter.join(features[node]))
            f.write(delimiter + target[node] + '\n')

    print(f'Created {target_processed_path}/{dataset_name}/{dataset_name}.cites and {target_processed_path}/{dataset_name}/{dataset_name}.content')


def transform_pubmed(inFeat, outFeat, inNet, outNet):
    with open(inFeat) as f:
        with open(outFeat, 'w') as out:
            ignore = True
            head = True
            target = {}
            features = {}
            for line in f:
                if ignore:
                    ignore = False
                    continue
                if head:
                    head = False
                    s = line.split()[1:-1]
                    s = [x.split(':')[1] for x in s]
                    feats_nums = {s[i]: i for i in range(len(s))}
                    continue
                s = line.split()
                id = s[0]
                target[id] = s[1].split('=')[1]
                assert (s[1].split('=')[0] == 'label')
                idx = [feats_nums[x.split('=')[0]] for x in s[2:-1]]
                feats = [1 if i in idx else 0 for i in range(len(feats_nums))]
                features[id] = feats
                out.write(id + ' ')
                out.write(' '.join([str(x) for x in feats]))
                out.write(' ' + target[id] + '\n')
    with open(inNet) as f:
        with open(outNet, 'w') as out:
            ignore = True
            head = True
            for line in f:
                if ignore:
                    ignore = False
                    continue
                if head:
                    head = False
                    continue
                s = line.split()
                u = s[1].split(':')[1]
                v = s[3].split(':')[1]
                out.write(v + ' ' + u + '\n')


def create_cora():
    dataset_name = 'cora'
    cora_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    target_raw = 'data/graphs/raw/cora.tgz'
    dir_path = 'data/graphs/raw'
    target_processed = 'data/graphs/processed'
    create_citation_dataset(dataset_name, cora_url, target_raw, dir_path, target_processed)


def create_citeseer():
    dataset_name = 'citeseer'
    citeseer_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz'
    target_raw = 'data/graphs/raw/citeseer.tgz'
    dir_path = 'data/graphs/raw'
    target_processed = 'data/graphs/processed'
    create_citation_dataset(dataset_name, citeseer_url, target_raw, dir_path, target_processed)


def create_texas():
    dataset_name = 'texas'
    texas_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/WebKB.tgz'
    target_raw = 'data/graphs/raw/WebKB.tgz'
    dir_path = 'data/graphs/raw'
    target_processed = 'data/graphs/processed'
    create_citation_dataset(dataset_name, texas_url, target_raw, dir_path, target_processed, raw_folder_name = 'WebKB',threshold = 16)

def create_washington():
    dataset_name = 'washington'
    washington_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/WebKB.tgz'
    target_raw = 'data/graphs/raw/WebKB.tgz'
    dir_path = 'data/graphs/raw'
    target_processed = 'data/graphs/processed'
    create_citation_dataset(dataset_name, washington_url, target_raw, dir_path, target_processed, raw_folder_name = 'WebKB',threshold = 16)

def create_wisconsin():
    dataset_name = 'wisconsin'
    wisconsin_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/WebKB.tgz'
    target_raw = 'data/graphs/raw/WebKB.tgz'
    dir_path = 'data/graphs/raw'
    target_processed = 'data/graphs/processed'
    create_citation_dataset(dataset_name, wisconsin_url, target_raw, dir_path, target_processed, raw_folder_name = 'WebKB',threshold = 10)

def create_cornell():
    dataset_name = 'cornell'
    cornell_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/WebKB.tgz'
    target_raw = '../../data/graphs/raw/WebKB.tgz'
    dir_path = '../../data/graphs/raw'
    target_processed = '../../data/graphs/processed'
    create_citation_dataset(dataset_name, cornell_url, target_raw, dir_path, target_processed, raw_folder_name = 'WebKB',threshold = 16)

def create_pubmed():
    dataset_name = 'pubmed'
    pubmed_url = 'https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz'
    dir_path = 'data/graphs/raw'
    target_tmp_file = dir_path + '/Pubmed-Diabetes.tgz'
    target_processed = 'data/graphs/processed'

    # download dataset
    if not os.path.exists(dir_path + '/' + dataset_name):
        print(f'Downloading {pubmed_url}...')
        wget.download(pubmed_url, target_tmp_file)
        tar = tarfile.open(target_tmp_file, "r:gz")
        tar.extractall(dir_path)
        tar.close()
        os.remove(target_tmp_file)
    print('File downloaded and extracted!')

    if not os.path.exists(f'{target_processed}/{dataset_name}'):
        os.mkdir(f'{target_processed}/{dataset_name}')

    transform_pubmed(f'{dir_path}/Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab',
                f'{target_processed}/{dataset_name}/{dataset_name}.content',
                f'{dir_path}/Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab',
                f'{target_processed}/{dataset_name}/{dataset_name}.cites')


def create_twitter():
    dataset_name = 'twitter'
    # need to download the .zip file from https://www.kaggle.com/manoelribeiro/hateful-users-on-twitter and place it in
    # 'data/graphs/raw/hateful-users-on-twitter.zip'
    target_raw = 'data/graphs/raw/hateful-users-on-twitter.zip'
    dir_path = 'data/graphs/raw'
    target_processed = 'data/graphs/processed'
    if not os.path.exists(dir_path + '/' + dataset_name):
        os.mkdir(dir_path + '/' + dataset_name)
    with zipfile.ZipFile(target_raw, 'r') as zip_ref:
        zip_ref.extractall(dir_path + '/' + dataset_name)
         # os.remove(target_raw)
    os.rename(dir_path + '/' + dataset_name + '/' + 'users.edges', dir_path + '/' + dataset_name + '/' + dataset_name + '.cites')
    df_users_neighborhood_anon = pd.read_csv(dir_path + '/' + dataset_name + '/users_neighborhood_anon.csv')
    print('loaded users_neighborhood_anon.csv')
    df_users_neighborhood_anon = df_users_neighborhood_anon[df_users_neighborhood_anon.hate != 'other']
    print('filtered annotated')
    cols = df_users_neighborhood_anon.columns.tolist()
    with open('data/graphs/raw/twitter_columns', 'r') as f:
        cols = [x[:-1] for x in f.readlines()]
    df_users_neighborhood_anon = df_users_neighborhood_anon.dropna()
    print(df_users_neighborhood_anon.shape)
    cols.append(cols.pop(cols.index('hate')))
    df_users_neighborhood_anon = df_users_neighborhood_anon[cols] * 1 # *1 to convert boolean to float
    print(f'filtered features, shape = {df_users_neighborhood_anon.shape}')

    for column in df_users_neighborhood_anon.columns:
        df_users_neighborhood_anon[column] = df_users_neighborhood_anon[column].replace("\t", " ")
    df_users_neighborhood_anon.to_csv(dir_path + '/' + dataset_name + '/' + dataset_name + '.content',
                                      index=False, sep="\t", quoting=csv.QUOTE_NONE, header=False)
    create_citation_dataset(dataset_name, None, None, dir_path, target_processed)

def merge_networks(directory, datasets, output_prefix):
    filenames = [f'{directory}/{x}.cites' for x in datasets]
    with open(f'{output_prefix}.cites', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    filenames = [f'{directory}/{x}.content' for x in datasets]
    with open(f'{output_prefix}.content', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def create_webkb():
    merge_networks('data/graphs/raw/WebKB', 'texas cornell washington wisconsin'.split(),'data/graphs/interim/webkb/webkb') 
    create_citation_dataset('webkb', None, None, dir_path = 'data/graphs/interim',
                            target_processed_path = 'data/graphs/processed',
                            raw_folder_name = 'webkb', threshold=25,select_largest_component=False)

def create_webkb_small():
    merge_networks('data/graphs/raw/WebKB', 'texas cornell washington wisconsin'.split(),'data/graphs/interim/webkb_small/webkb_small') 
    create_citation_dataset('webkb_small', None, None, dir_path = 'data/graphs/interim',
                            target_processed_path = 'data/graphs/processed',
                            raw_folder_name = 'webkb_small', threshold=25,select_largest_component=False, sample_features=50)

def create_jigsaw_graph(filename):

    # WARNING: the graph can have several connected components
    df = pd.read_csv(f'data/graphs/raw/jigsaw/{filename}.csv')
    features = [[s.strip() for s in x[1:-1].split(',')] for x in df.intermediate.values]
    labels = df.toxic.values
    neighbors = [[s.strip() for s in x[1:-1].split(',')] for x in df.knn.values]
    edges = [[str(i),j] for i in df.id.values for j in neighbors[i]]
    if not os.path.exists(f'data/graphs/processed/jigsaw_{filename}'):
        os.mkdir(f'data/graphs/processed/jigsaw_{filename}')
    with open(f'data/graphs/processed/jigsaw_{filename}/jigsaw_{filename}.cites','w') as outfile:
        for edge in edges:
            outfile.write(' '.join(edge)+'\n')
        
    with open(f'data/graphs/processed/jigsaw_{filename}/jigsaw_{filename}.content','w') as outfile:
        for i in df.id.values:
            outfile.write(str(i) + ' ' + ' '.join(features[i]) + ' ' +str(labels[i]) + '\n')




def create_wiki_cs():
    dataset_name = 'wiki_cs'
    url = 'https://github.com/pmernyei/wiki-cs-dataset/blob/master/dataset/data.json?raw=true'
    dir_path = 'data/graphs/raw'
    target_file = dir_path + '/' + dataset_name + '/data.json'
    target_processed = 'data/graphs/processed'

    raw_folder_name = dataset_name

    # download dataset
    if (url is not None) and (not os.path.exists(dir_path + '/' + dataset_name)):
        os.mkdir(dir_path + '/' + dataset_name)
        print(f'Downloading {url}...')
        wget.download(url, target_file)
        print(f'File downloaded at {target_file}!')
    else: print(f'Directory downloaded at {dir_path}/{dataset_name} already exists!')

    if not os.path.exists(f'{target_processed}/{dataset_name}'):
        os.mkdir(f'{target_processed}/{dataset_name}')


    graph = json.load(open (target_file, "r"))

    features = graph['features']
    labels = graph['labels']
    links = graph['links']

    with open(f'{target_processed}/{dataset_name}/{dataset_name}.cites','w') as outfile:
        for u in range(len(links)):
            for v in links[u]:
                outfile.write(str(u) + ' ' + str(v) + '\n')

    with open(f'{target_processed}/{dataset_name}/{dataset_name}.content','w') as outfile:
        for u in range(len(features)):
            outfile.write(str(u) + ' ' + ' '.join([str(f) for f in features[u]]) + ' ' + str(labels[u]) + '\n')





if __name__ == '__main__':
    create_wiki_cs()
    # create_jigsaw_graph('validation_knn_40')
    # create_webkb_small()
    #create_cornell()
    #create_wisconsin()
    #create_washington()
    #create_texas()
    # create_twitter()
