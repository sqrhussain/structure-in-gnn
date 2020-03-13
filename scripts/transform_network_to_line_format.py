import argparse
import os
def transform_line(inFile,outFile,undirected=True):
    with open(outFile,'w') as out:
        with open(inFile,'r') as f:
            for line in f:
                out.write('\t'.join(line.split()) + '\t1\n')
                if not undirected:
                    continue
                out.write('\t'.join(reversed(line.split())) + '\t1\n')

def reverse_direction_line(inFile,outFile):
    with open(outFile,'w') as out:
        with open(inFile,'r') as f:
            for line in f:
                out.write('\t'.join(reversed(line.split())) + '\t1\n')
                

parser = argparse.ArgumentParser(description="Convert a network into line format.")
parser.add_argument('--dataset',
                    default="cora",
                    help='Dataset name. Default is cora.')
                    
args = parser.parse_args()

if not os.path.exists(f'data/graphs/line_format/{args.dataset}'):
    os.mkdir(f'data/graphs/line_format/{args.dataset}')

transform_line(f'data/graphs/processed/{args.dataset}/{args.dataset}.cites',
                f'data/graphs/line_format/{args.dataset}/{args.dataset}_undirected.cites',
                True)
                
transform_line(f'data/graphs/processed/{args.dataset}/{args.dataset}.cites',
                f'data/graphs/line_format/{args.dataset}/{args.dataset}_directed.cites',
                False)
                
reverse_direction_line(f'data/graphs/processed/{args.dataset}/{args.dataset}.cites',
                f'data/graphs/line_format/{args.dataset}/{args.dataset}_reversed.cites')
