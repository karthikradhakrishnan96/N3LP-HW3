import csv
import json
import numpy as np
from sklearn.metrics import f1_score


class Priori():
    def __init__(self, tree_file):
        f = open(tree_file, 'r')
        self.tree = json.load(f)
        self.tree = self.tree['twitter']['dev']
        self.A = np.array([[468, 1703, 206, 227, 0], [125, 949, 88, 101, 0], [28, 129, 33, 17, 0], [12, 115, 8, 13, 0], [277, 11, 9, 0, 0]])
        self.A = self.A / self.A.sum(axis=1)[:, np.newaxis]
        self.mapp = {"None": 4, "support": 0, "comment": 1, "deny": 2, "query": 3}
        self.rev_mapp = {self.mapp[x]: x for x in self.mapp}
        f.close()

    def get_prior_f1(self, lines, trans_weight):
        self.updated_results = {} # bad but okay
        self.trans_weight = trans_weight
        bert_results = {}
        for line in lines[1:]:
            if not line[0].isdigit():
                pass
            bert_results[line[0]] = [line[1], line[2], np.array((line[3]))]
        self.bert_results = bert_results
        for thread in self.tree:
            self.do_thread(thread)
        preds = []
        reals = []
        for pred, real in self.updated_results.values():
            preds.append(pred)
            reals.append(real)

        f1 = f1_score(reals, preds, average='macro').item()
        return f1

    def do_thread(self, thread):
        all = {}
        all[str(thread['source']['id_str'])] = thread['source']
        for reply in thread['replies']:
            all[str(reply['id_str'])] = reply
        struct = thread['structure'][thread['source']['id_str']]
        self.do_single(thread['source'], 0, struct, all)

    def do_single(self, tweet, depth, struct, all, parent_label="None"):
        my_label_vs = self.bert_results[tweet['id_str']]
        my_pred = my_label_vs[0]
        my_REAL = my_label_vs[1]
        my_conf = my_label_vs[2]
        transition = self.A[self.mapp[parent_label]][:-1]
        new_conf = my_conf + (self.trans_weight * transition if tweet['id_str'].isdigit() else 0)
        # new_conf = my_conf
        new_best_arg = new_conf.argmax()

        self.updated_results[tweet['id_str']] = [self.rev_mapp[new_best_arg], my_REAL]

        if type(struct) != type({}):
            return
        for key in struct:
            if key not in all:
                # print("    " * (depth + 1) + "~" * 10 + "SKIPPED" + "~" * 10)
                continue
            self.do_single(all[key], depth + 1, struct[key], all, self.rev_mapp[new_best_arg])
        return

