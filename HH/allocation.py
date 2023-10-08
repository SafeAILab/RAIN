import argparse
parser = argparse.ArgumentParser(description='all')
parser.add_argument('--outdir', type=str, default='outdir')
parser.add_argument('--modelname', type=str, default='modelname')
parser.add_argument('--nump', type=int, default=8)
parser.add_argument('--s', type=int)
parser.add_argument('--e', type=int)

args = parser.parse_args()

import os
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
s=args.s
e=args.e
num_p=args.nump
def split_range(start, end, n,over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n   # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append((previous, previous + current_interval - 1))  # '-1' because the end is inclusive
        previous += current_interval

    return intervals

def run_command(cmd):
    os.system(cmd)


if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
if not os.path.exists('{}_dicv'.format(args.outdir)):
    os.makedirs('{}_dicv'.format(args.outdir))

gpu_a=split_range(0,7,num_p)
data_a=split_range(s,e,num_p,over=True)
commands=[]
for i in range(num_p):
    index=i
    start=data_a[i][0]
    end=data_a[i][1]
    # gpu_index_str = [str(i) for i in gpu_index]
    # gpu_index_str=','.join(gpu_index_str)
    gpu_index = list(range(gpu_a[i][0], gpu_a[i][1] + 1))
    gpu_index_str = ' '.join(map(str, gpu_index))
    #gpu_index_str='['+gpu_index_str+']'
    command="python main.py --start={} --end={} --index={} --gpu_index {} --outdir {} --modelname {}".format(start,end,index,gpu_index_str,args.outdir,args.modelname)
    commands.append(command)
# run_command(commands[0])
#commands=commands[:1]
with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
