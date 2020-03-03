import wget
import tarfile
import os
import networkx as nx
import scipy



def create_citation_dataset(dataset_name, url, target_tmp_file, dir_path, target_processed_path,
                            threshold=50):  # to use for CORA and CITESEER

    # download dataset
    if not os.path.exists(dir_path + '/' + dataset_name):
        print(f'Downloading {url}...')
        wget.download(url, target_tmp_file)
        tar = tarfile.open(target_tmp_file, "r:gz")
        tar.extractall(dir_path)
        tar.close()
        os.remove(target_tmp_file)
    print('File downloaded and extracted!')

    # load features
    features_path = f'{dir_path}/{dataset_name}/{dataset_name}.content'
    edges_path = f'{dir_path}/{dataset_name}/{dataset_name}.cites'
    target = {}
    features = {}
    class_count = {}
    nodes = set()
    with open(features_path) as f:
        for line in f:
            info = line.split()
            if len(info) == 1:
                info = line.split(',')
            nodes.add(info[0])
            y = info[-1]
            target[info[0]] = y
            features[info[0]] = info[1:-1]

            if y in class_count:
                class_count[y] += 1
            else:
                class_count[y] = 1

    # remove nodes with rare labels
    nodes = {u for u in nodes if class_count[target[u]] >= threshold}

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

    # select biggest connected component
    undirected_graph = nx.Graph()  # undirected, just to find the biggest connected component
    undirected_graph.add_edges_from(edges)
    largest_component = max(nx.connected_component_subgraphs(undirected_graph), key=len)

    nodes = largest_component.nodes
    if not os.path.exists(f'{target_processed_path}/{dataset_name}'):
        os.mkdir(f'{target_processed_path}/{dataset_name}')
    edges = [e for e in edges if e[0] in nodes and e[1] in nodes]
    with open(f'{target_processed_path}/{dataset_name}/{dataset_name}.cites', 'w') as f:
        for edge in edges:
            f.write(' '.join(edge) + '\n')

    with open(f'{target_processed_path}/{dataset_name}/{dataset_name}.content', 'w') as f:
        for node in nodes:
            f.write(node + ' ')
            f.write(' '.join(features[node]))
            f.write(' ' + target[node] + '\n')

    print(f'Created {dataset_name}.cites and {dataset_name}.content')


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

if __name__ == '__main__':
    create_cora()
