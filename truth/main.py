# coding=utf-8
import argparse

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0, 1])
parser.add_argument('--outdir', type=str, default='outdir')
parser.add_argument('--modelname', type=str, default='modelname')

args = parser.parse_args()
maxlen = 2048
maxT = 50
minT = 16
Vt = 0.8
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from sentence_transformers import SentenceTransformer


def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(0)

from nodes import node


def save_dict(dict_input, filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            dict_existing = json.load(file)
        dict_merged = {**dict_existing, **dict_input}
    else:
        dict_merged = dict_input

    with open(filename, 'w') as file:
        json.dump(dict_merged, file)


def find_all_indices(text, substring):
    indices = []
    start_index = 0
    while True:
        index = text.find(substring, start_index)
        if index == -1:
            break
        indices.append(index)
        start_index = index + 1
    return indices


with open('f2.txt') as f:
    fsred = f.read()
with open('r1.txt') as f:
    redA = f.read()
with open('r2.txt') as f:
    redB = f.read()
outdir = args.outdir
try:
    with open('{}/res_{}.json'.format(outdir, args.index)) as f:
        res = json.loads(f.read())
    qs = []
    for i in res:
        qs.append(i['question'])
except:
    res = []
    qs = []


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    ds = load_dataset('truthful_qa', 'generation')
    ds = ds['validation']
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    # ds1 = ds.select(range(100,200))
    # dst=ds.select(range(200,300))
    # ds2=ds.select(range(300,len(ds)))
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
            "queryf": [],
            'best_answer':[],
            'source':[]
        }
        for i in range(len(examples['question'])):
            transcript = examples['question'][i]
            text = transcript
            tokenized_question = tokenizer(text, truncation=True)
            new_examples["input_ids"].append(tokenized_question["input_ids"])
            # textf = fschat + '\n\nQ: ' + text + '\nA:'
            textf = text
            new_examples["query"].append(textf)
            new_examples["queryf"].append(textf)
            new_examples["best_answer"].append(examples['best_answer'][i])
            new_examples["source"].append(examples['source'][i])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        # num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    ds1 = ds1.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds1.set_format(type="torch")

    return ds1


if not os.path.exists(outdir):
    os.makedirs(outdir)
if not os.path.exists('{}_dicv'.format(outdir)):
    os.makedirs('{}_dicv'.format(outdir))
modelname = args.modelname
print(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForCausalLM.from_pretrained(
    modelname,
    # load_in_4bit=True,
    load_in_8bit=True,
    # torch_dtype=torch.float16,
    device_map="auto",
)
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
testdataset = build_dataset_rank(tokenizer)
testloader = DataLoader(testdataset, batch_size=1, shuffle=False)


@torch.no_grad()
def genc(s, model, tokenizer):
    '''
    vanilla autoregression
    '''
    input_ids = tokenizer(s, return_tensors="pt").input_ids
    outs = model.generate(inputs=input_ids.cuda(), max_length=maxlen, use_cache=False)
    outstr = tokenizer.decode(outs[0], skip_special_tokens=True)
    return outstr


model.eval()

def remove_after_last_period(s):
    head, sep, tail = s.rpartition('.')
    return head+'.' if sep else s

@torch.no_grad()
def getv(getoken, model, tokenizer, dic, dicp, maxlen):
    '''
    score through self-evaluation
    '''
    text, simgstate = simg(dicp, getoken, model, tokenizer, maxlen)
    inds = find_all_indices(text, '[INST]')
    if len(inds) > 1:
        text = text[:inds[1]]
    text = text.replace('[INST] ', 'Q: ')
    text = text.replace(' [/INST] ', '\nA:')
    text = remove_after_last_period(text)
    text = text.strip()

    if text not in dic:
        text = "[INST] " + fsred + '\n\n' + text + '\nIs the answer truthful? [\INST] '
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        outs = model(input_ids.cuda())
        logits = outs.logits
        last_token_logits = logits[0, -1, :]
        prob = F.softmax(last_token_logits.float(), dim=0)
        p_A = max(prob[3869].item(), prob[8241].item())
        p_B = max(prob[1939].item(), prob[3782].item())
        if p_A > p_B:
            A = 1
        else:
            A = 0
        v = A
        v = (v - 0.5) * 2
        dic[text] = v
    else:
        v = dic[text]
    return v, simgstate, len(simgstate) - len(getoken)


@torch.no_grad()
def simg(dicp, orstate, model, tokenizer, maxlen=1280):
    '''
    simulation generation for more accurate self-evaluation
    '''
    state = copy.deepcopy(orstate)
    past_key_values = None
    while 1:
        if len(state) > maxlen:
            break
        if tokenizer.eos_token_id in state:
            break
        probs, past_key_values = getp(state, model, dicp, topk=-1, return_past_key_values=True,
                                      past_key_values=past_key_values)
        token = int(torch.multinomial(probs, num_samples=1))
        state.append(token)
        if token == tokenizer.eos_token_id:
            break
    tmpstr = tokenizer.decode(state, skip_special_tokens=True)
    return tmpstr, state


def prepare_logits_processor(
        temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.no_grad()
def getp(state, model, dicp, topk=-1, topp=1.0, temperature=1.0, repetition_penalty=0.0, return_last_logits=False,
         return_past_key_values=False, past_key_values=None):
    '''
    query LLM
    '''
    if tuple(state) not in dicp:
        if past_key_values != None:
            input_ids = torch.tensor([[state[-1]]])
            outs = model(input_ids.cuda(), past_key_values=past_key_values)
        else:
            input_ids = torch.tensor([state])
            outs = model(input_ids.cuda())
        logits = outs.logits
        past_key_values = outs.past_key_values
        last_logits = logits[:, -1, :].float().cpu()
        dicp[tuple(state)] = last_logits
    else:
        last_logits = dicp[tuple(state)]
        past_key_values = None

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, topp, topk
    )
    last_token_logits = logits_processor(None, last_logits)[0]
    probs = torch.softmax(last_token_logits, dim=-1)
    if return_last_logits and return_past_key_values:
        return probs, last_logits, past_key_values
    if return_last_logits:
        return probs, last_logits
    if return_past_key_values:
        return probs, past_key_values
    return probs



@torch.no_grad()
def group_getp(state, model, dicp, topk=10, maxnew=10):
    '''
    group query LLM
    '''
    outs = []
    outsset = []
    etmpp = []
    if maxnew == 1:
        p, last_logits = getp(state, model, dicp, topk=topk, return_last_logits=True)
        acp = p.cpu().detach().squeeze(0).numpy()
        legal = np.where(acp > 0)[0]
        acp = acp[legal]
        acp = zip(legal, acp)
        for ac, p in acp:
            outs.append(([ac], p))
        return outs, last_logits

    greedytmpstate = copy.deepcopy(state)
    greedytmplog = torch.tensor(0.0)
    greedytmptokens = []
    greedy_past_key_values = None
    for i in range(maxnew):
        greedyprobs,greedy_past_key_values = getp(greedytmpstate, model, dicp, topk=15,return_past_key_values=True,past_key_values=greedy_past_key_values)
        greedytoken = int(torch.argmax(greedyprobs))
        greedylogp = torch.log(greedyprobs[greedytoken])
        greedytmplog += greedylogp
        greedytmptokens.append(greedytoken)
        greedytmpstate.append(greedytoken)
    outsset.append(greedytmptokens)


    for _ in range(topk - 1):
        tmpstate = copy.deepcopy(state)
        tmplog = torch.tensor(0.0)
        tmptokens = []
        past_key_values = None
        for i in range(maxnew):
            probs,past_key_values = getp(tmpstate, model, dicp, topk=15,return_past_key_values=True,past_key_values=past_key_values)
            token = int(torch.multinomial(probs, num_samples=1))
            logp = torch.log(probs[token])
            tmplog += logp
            tmptokens.append(token)
            tmpstate.append(token)
        if tmptokens not in outsset:
            outsset.append(tmptokens)
            tmpp = torch.exp(tmplog)
            outs.append((tmptokens, tmpp.item()))
            etmpp.append(tmpp.item())
        if len(outs) >= topk - 1:
            break



    greedytmpp = torch.exp(greedytmplog)
    if len(etmpp)>0:
        etmpp=np.array(etmpp)
        greedytmpp = min(greedytmpp.item(), etmpp.sum())
        greedytmpp = max(greedytmpp, etmpp.max()+etmpp.min())
    else:
        greedytmpp=greedytmpp.item()
    outs = [(greedytmptokens, greedytmpp)] + outs

    return outs


def node2dic(node, state, tokenizer):
    d = {}
    dd = {}
    tmpstr = tokenizer.decode(state, skip_special_tokens=True)
    for act, node in node.children.items():
        actstr = tokenizer.decode(act, skip_special_tokens=True)
        n = node.n
        q = node.q
        dd[actstr] = (n, q)
    d[tmpstr] = dd
    return d


def getmaxnew(step):
    '''
    return the length of token set
    '''
    if step == 0:
        return 1
    if step == 1:
        return 2
    if step == 2:
        return 4
    return 10

def slice_until_2(tup):
    result = []
    for item in tup:
        result.append(item)
        if item == 2:
            break
    return tuple(result)

@torch.no_grad()
def search(root, state, model, tokenizer, dic, dicp, maxlen=1024):
    state = copy.deepcopy(state)
    cnode = root
    reward = 0
    action = (-1,-1)

    while not cnode.isleaf():
        addflag = cnode.checkadd()
        if addflag:
            maxnew = getmaxnew(cnode.step)
            agp = group_getp(state, model, dicp, topk=2, maxnew=maxnew)
            cnode.add(agp)
        action, cnode = cnode.select()
        state.extend(action)

    # check whether the generation is finished
    if len(state) > maxlen or action == tokenizer.eos_token_id or tokenizer.eos_token_id in action:
        v, embeding_token, path_n = getv(state, model, tokenizer, dic, dicp, maxlen)
    else:
        v, embeding_token, path_n = getv(state, model, tokenizer, dic, dicp, maxlen)
        maxnew = getmaxnew(cnode.step)
        if maxnew == 1:
            gp, egp = group_getp(state, model, dicp, topk=10, maxnew=maxnew)
        else:
            gp = group_getp(state, model, dicp, topk=10, maxnew=maxnew)
            egp = copy.deepcopy(gp)

        p = [i[1] for i in gp]
        act = [i[0] for i in gp]
        acp = np.array(p)
        acp = acp / acp.sum()

        if cnode.parent == None:
            acp = 0.75 * acp + 0.25 * np.ones(len(acp)) / len(acp)
            acp = acp / acp.sum()
        acp = zip(act, acp)
        cnode.expand(root=root, ac_p=acp, reward=reward, state=state, logits=egp)
    cnode.backup(v, embeding_token, tokenizer, encoder, path_n=path_n)


@torch.no_grad()
def gmeval(batch, model, tokenizer):
    '''
    outer loop
    '''
    dic, dicp = {}, {}
    query = batch['query'][0]
    query = "[INST] " + query + " [/INST] "
    # answer = batch['input_ids'][0]
    best_answer=batch['best_answer'][0]
    source=batch['source'][0]


    if query in qs:
        return None
    instrcot = query
    input_ids = tokenizer(instrcot, return_tensors="pt").input_ids
    slen = input_ids.shape[1]
    state = input_ids.tolist()[0]


    root = node(root=None, parent=None, prior_p=0, step=0)


    initi = 0
    while 1:
        for i in range(initi, max(maxT, initi + 15)):
            search(root, state, model, tokenizer, dic, dicp, maxlen=maxlen)
            try:
                bq, bfn = root.get_max_nq_value()
            except:
                bq, bfn = 0, 0
            if bfn > minT and bq > Vt:
                break

        try:

            move = root.get_max_n_action()
            rootd = node2dic(root, state, tokenizer)
            save_dict(rootd, '{}_dicv/res_root_{}.json'.format(outdir, args.index))

            move=slice_until_2(move)
            state.extend(move)
            oroot = root
            root = root.children[move]
            root.parent = None
            root.minqn = oroot.minqn
            root.maxqn = oroot.maxqn
            cp = [root.children[i].p for i in root.children]
            cp = np.array(cp)
            cp = 0.75 * cp + 0.25 * np.ones(len(cp)) / len(cp)
            cp = cp / cp.sum()
            for id, i in enumerate(root.children):
                root.children[i].p = cp[id]
            initi = root.fn
        except:
            move = tokenizer.eos_token_id



        if len(state) > maxlen:
            break
        if move == tokenizer.eos_token_id:
            break
        if tokenizer.eos_token_id in move:
            break

    raina = tokenizer.decode(state, skip_special_tokens=True)
    raina= raina.replace('[INST] ', 'Q: ')
    raina = raina.replace(' [/INST] ', '\nA:')
    raina = raina.strip()



    pa = genc(query, model, tokenizer)
    pa = pa.replace('[INST] ', 'Q: ')
    pa = pa.replace(' [/INST] ', '\nA:')
    pa = pa.strip()




    tmp = {'question': query, 'raina': raina, 'pa': pa,'source':source,'best_answer':best_answer}
    save_dict(dic, '{}_dicv/res_{}.json'.format(outdir, args.index))
    return tmp


for epoch_test, batch_test in tqdm(enumerate(testloader)):
    tmp = gmeval(batch_test, model, tokenizer)
    if tmp != None:
        res.append(tmp)
    with open('{}/res_{}.json'.format(outdir, args.index), 'w') as f:
        f.write(json.dumps(res))
