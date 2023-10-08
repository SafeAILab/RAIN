import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

torch.set_num_threads(1)

gamma = 1.0


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def prepare_logits_processor(
        temperature=1.0, repetition_penalty=0.0, top_p=1.0, top_k=-1
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()

    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list

class node():


    def __init__(self, root, parent, prior_p, tokens=None,self_embeding=None,step=0):
        self.step=step
        self.root = root
        self.parent = parent
        self.tokens=tokens
        self.children = {}
        self.n = 0
        self.fn = 0
        self.q = 0
        self.u = 0
        self.reward = 0
        self.p = prior_p
        self.embeding = self_embeding
        self.self_embeding = self_embeding
        self.cosine_similarities = []
        if parent == None:
            self.root = self
            self.maxqn = 1
            self.minqn = -1


    def get_max_n_action(self):
        if not self.children:
            return None

        max_n_value = max(child.n for child in self.children.values())
        max_n_nodes = [(ac, child) for ac, child in self.children.items() if child.n == max_n_value]
        best_ac_node = max(max_n_nodes, key=lambda x: x[1].q)

        return best_ac_node[0]

    def get_max_nq_value(self):
        best_action = self.get_max_n_action()
        if best_action is None:
            return None
        return self.children[best_action].q,self.children[best_action].fn

    def expand(self, root, ac_p, reward,state=None,logits=None):
        self.reward = reward
        self.logits=logits
        self.state=state
        for ac, p in ac_p:
            self.children[tuple(ac)] = node(root=root, parent=self, prior_p=p,tokens=tuple(ac),step=self.step+1)

    def child_embedings_variance(self):
        if self.isleaf():
            return None
        else:
            child_embedings = []
            for child in self.children.values():
                if child.embeding is None:
                    return None
                else:
                    child_embedings.append(child.embeding)

            child_embedings = torch.stack(child_embedings)
            variance = torch.var(child_embedings, dim=0)
            return variance

    def getqu(self):
        cpuct = 2
        if self.n == 0:
            qh = 0
        else:
            qh = (self.q - self.root.minqn) / (self.root.maxqn - self.root.minqn + 1e-5)

        self.u = cpuct * self.p * np.sqrt(self.parent.n) / (1 + self.n)

        return qh + self.u

    def num_children(self):
        return len(self.children)

    def max_child_q(self):
        if self.isleaf():
            return None
        else:
            child_qs = [child.q for child in self.children.values()]
            return max(child_qs)

    def checkadd(self):

        c = next(iter(self.children.values()))
        the_norm = 0.03 * len(c.tokens)
        the_q = 0.0
        nc=self.num_children()
        if nc<20:
            embeding_var=self.child_embedings_variance()
            if embeding_var is not None:
                norm=torch.norm(embeding_var)
                max_q=self.max_child_q()
                if norm<the_norm and max_q<the_q:
                    c=next(iter(self.children.values()))
                    if len(c.tokens)==1:
                        logits_processor = prepare_logits_processor(
                            top_k=self.num_children() + 1
                        )
                        last_token_logits = logits_processor(None, self.logits)[0]
                        probs = torch.softmax(last_token_logits, dim=-1)
                        acp = probs.detach().squeeze(0).numpy()
                        legal = np.where(acp > 0)[0]
                        acp = acp[legal]
                        acp = acp / acp.sum()
                        if self.isroot():
                            acp = 0.75 * acp + 0.25 * np.ones(len(acp)) / len(acp)
                            acp = acp / acp.sum()
                        ac_p = zip(legal, acp)
                        for ac, p in ac_p:
                            if tuple([ac]) not in self.children:
                                self.children[tuple([ac])] = node(root=self.root, parent=self, prior_p=p,
                                                                  tokens=tuple([ac]),
                                                                  step=self.step + 1)
                        return False
                    else:
                        return True
        return False

    def add(self,agp):
        self.logits=agp+self.logits
        gp = self.logits[:self.num_children() + 1]
        p = [i[1] for i in gp]
        act = [i[0] for i in gp]
        acp = np.array(p)
        acp = acp / acp.sum()

        if self.isroot():
            acp = 0.75 * acp + 0.25 * np.ones(len(acp)) / len(acp)
            acp = acp / acp.sum()
        ac_p = zip(act, acp)
        for ac, p in ac_p:
            if tuple(ac) not in self.children:
                self.children[tuple(ac)] = node(root=self.root, parent=self, prior_p=p, tokens=tuple(ac),step=self.step+1)

    def select(self):

        return max(self.children.items(), key=lambda act_node: act_node[1].getqu())

    def backup(self, v, state, tokenizer, encoder, path_n=0):
        sim_gamma = 0.2
        sim_the = 0.95
        g = (gamma) * v + self.reward
        self.n += 1
        self.fn += 1
        if self.parent:
            path_n += len(self.tokens)
            path_state = state[-path_n:]
            path_text = tokenizer.decode(path_state, skip_special_tokens=True)
            path_embeding = encoder.encode(path_text)
            path_embeding = torch.tensor(path_embeding)
            if self.embeding != None:
                self.embeding += (path_embeding - self.embeding) / self.fn
            else:
                self.embeding = path_embeding
            sims = self.calculate_cosine_similarity(path_embeding)
            for similarity, is_self, snode in sims:
                if is_self or similarity < sim_the:
                    continue
                similarity = similarity * sim_gamma
                before_n = snode.n
                snode.n += similarity
                snode.q = (snode.q * before_n + similarity * g) / snode.n

            self.parent.backup(g, state, tokenizer, encoder, path_n)

        self.q += (g - self.q) / self.n
        if not self.isroot():
            self.root.minqn = min(self.root.minqn, self.q)
            self.root.maxqn = max(self.root.maxqn, self.q)



    def calculate_cosine_similarity(self, path_embeding):
        result = []
        siblings = [child for child in self.parent.children.values()]
        for sibling in siblings:
            if sibling.embeding==None:
                similarity=0.0
            else:
                similarity = cosine_similarity(self.embeding.unsqueeze(0).float(),
                                               sibling.embeding.unsqueeze(0).float())
                similarity=similarity.item()
            is_self = (sibling is self)
            result.append((similarity, is_self, sibling))
        return result

    def isleaf(self):
        return self.children == {}

    def isroot(self):
        return self.parent is None
