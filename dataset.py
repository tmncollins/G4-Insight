import torch
from torch.utils.data import Dataset
import numpy as np
from encoding import nt_embedding

class G4_Dataset(Dataset):
    def __init__(self, data, encoding, context=None, debug=False):
        self.ids = []
        self.up_seqs = []
        self.down_seqs = []
        self.targets = []
        self.pqs = []
        self.chroms = []

        idx = 0
        for line in data:
            up, pqs, down, chrom, target = line.split(",")
            target = int(target)
            self.ids.append(idx)
            self.chroms.append(chrom)
            if context == None:
                self.up_seqs.append(encoding(up))
                self.down_seqs.append(encoding(down))
            else:
                self.up_seqs.append(torch.tensor(encoding(up[len(up)-context:])).float())
                self.down_seqs.append(torch.tensor(encoding(down[:context])).float())
            self.pqs.append(torch.tensor(encoding(pqs)).float())
            self.targets.append([target])

            idx += 1

        print(len(self.up_seqs), len(self.down_seqs), len(self.targets), len(self.pqs))
        self.targets = torch.tensor(np.array(self.targets)).float()

    def set_context(self, context):
        return

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        id = self.ids[index]
        upseq = self.up_seqs[index]
        pqs = self.pqs[index]
        downseq = self.down_seqs[index]
        target = self.targets[index]

        return id, upseq, pqs, downseq, target
    
    def filter_chroms(self, chrom):
        t, u, p, d, i = [], [], [], [], [] 
        for j in range(len(self.targets)):
            if self.chroms[j] == chrom:
                t.append(self.targets[j])
                u.append(self.up_seqs[j])
                p.append(self.pqs[j])
                d.append(self.down_seqs[j])
                i.append(self.ids[j])
        self.targets = t
        self.up_seqs = u
        self.pqs = p
        self.down_seqs = d
        self.ids = i
