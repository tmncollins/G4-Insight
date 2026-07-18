import torch.nn as nn
import torch.nn.functional as F
import torch

class G4_args():

    def __init__(self):
        self.input_size = 0
        self.rnn_layer_size = 0
        self.rnn_layer_n = 0
        self.rnn_dropout = 0
        self.rnn_bidirectional = True
        self.pool_size = 0
        self.mlp_layer_size = 0
        self.mlp_dropout = 0
        self.vocab_size = 0
        self.cnn_kernel_size = 0
        self.cnn_channels_in = 0
        self.cnn_channels_out = 0
        self.context = 0
        self.L = 0


    def save(self, file_path, model_name):
        VALUES = {"input_size": self.input_size, "rnn_layer_n": self.rnn_layer_n, "rnn_layer_size": self.rnn_layer_size, 
            "rnn_dropout": self.rnn_dropout, "rnn_bidirectional": self.rnn_bidirectional, "pool_size": self.pool_size,
            "mlp_layer_size": self.mlp_layer_size, "mlp_dropout": self.mlp_dropout, "vocab_size": self.vocab_size,
            "cnn_kernel_size": self.cnn_kernel_size, "cnn_channels_in": self.cnn_channels_in, "cnn_channels_out": self.cnn_channels_out,
            "context": self.context, "L":self.L}
        f = open(file_path + model_name + "_args.txt", "w")
        text = ""
        for name, val in VALUES.items():
            text += name + "," + str(val) + "\n"
        f.write(text)
        f.close()

    def load(self, file_path, model_name):
        f = open(file_path + model_name + "_args.txt", "r").read().split("\n")
        for line in f:
            try:
                name, val = line.strip().split(",")
                
                if "." in val: val = float(val)
                elif "True" in val: val = True
                elif "False" in val: val = False
                else: val = int(val)

                if   name == "input_size": self.input_size = val
                elif name == "rnn_layer_n": self.rnn_layer_n = val
                elif name == "rnn_layer_size": self.rnn_layer_size = val
                elif name == "rnn_dropout": self.rnn_dropout = val
                elif name == "rnn_bidirectional": self.rnn_bidirectional = val
                elif name == "pool_size": self.pool_size = val
                elif name == "mlp_layer_size": self.mlp_layer_size = val
                elif name == "mlp_dropout": self.mlp_layer_dropout = val
                elif name == "vocab_size": self.vocab_size = val
                elif name == "cnn_kernel_size": self.cnn_kernel_size = val
                elif name == "cnn_channels_in": self.cnn_channels_in = val
                elif name == "cnn_channels_out": self.cnn_channels_out = val
                elif name == "context": self.context = val
                elif name == "L": self.L = val
            except:
                pass

class G4_Meta():
    def __init__(self, models, threshold):
        self.models = models
        self.threshold = threshold
        self.sigmoid = nn.Sigmoid()
        
    def get_votes(self, _u, _pqs, _d):
        yes = 0
        no = 0
        for model in self.models:
            p = self.sigmoid(model(_u, _pqs, _d)).item()
            p = (p > self.threshold)
            if p: yes += 1
            else: no += 1
        return yes, no
    
    def predict(self, upseq, pqs, downseq):
        yes, no = self.get_votes(upseq, pqs, downseq)
        return yes > no

class G4_RNN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embed = nn.Embedding(args.vocab_size, args.cnn_channels_in)
        self.cnn = nn.Conv1d(args.cnn_channels_in, args.cnn_channels_out, args.cnn_kernel_size)
        self.rnn = nn.GRU(args.cnn_channels_out, args.rnn_layer_size, args.rnn_layer_n, dropout=args.rnn_dropout, bidirectional=args.rnn_bidirectional)
        D = 2 if args.rnn_bidirectional else 1
        self.mlp = nn.Sequential(
            nn.Linear(args.rnn_layer_size, args.mlp_layer_size),
            nn.Dropout(args.mlp_dropout),
            nn.SiLU(),
            nn.Linear(args.mlp_layer_size, args.mlp_layer_size),
            nn.Dropout(args.mlp_dropout),
            nn.SiLU(),
            nn.Linear(args.mlp_layer_size, 1)
        )

    def forward(self, up_seq, pqs, down_seq):
        tensor = torch.cat((up_seq, pqs, down_seq))
        tensor = self.embed(tensor.int())
        tensor = torch.transpose(tensor, 1, 0)
        tensor = self.cnn(tensor)
        tensor = torch.transpose(tensor, 1, 0)
        _, x = self.rnn(tensor)
        out = self.mlp(x[0])
        return out

class G4_RNN_ReLU(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embed = nn.Embedding(args.vocab_size, args.cnn_channels_in)
        self.cnn = nn.Conv1d(args.cnn_channels_in, args.cnn_channels_out, args.cnn_kernel_size)
        self.rnn = nn.GRU(args.cnn_channels_out, args.rnn_layer_size, args.rnn_layer_n, dropout=args.rnn_dropout, bidirectional=args.rnn_bidirectional)
        D = 2 if args.rnn_bidirectional else 1
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(args.rnn_layer_size, args.mlp_layer_size),
            nn.Dropout(args.mlp_dropout),
            nn.ReLU(),
            nn.Linear(args.mlp_layer_size, args.mlp_layer_size),
            nn.Dropout(args.mlp_dropout),
            nn.ReLU(),
            nn.Linear(args.mlp_layer_size, 1)
        )

    def forward(self, up_seq, pqs, down_seq):
        tensor = torch.cat((up_seq, pqs, down_seq))
        tensor = self.embed(tensor.int())
        tensor = torch.transpose(tensor, 1, 0)
        tensor = self.cnn(tensor)
        tensor = torch.transpose(tensor, 1, 0)
        tensor = self.relu(tensor)
        _, x = self.rnn(tensor)
        x = self.relu(x)
        out = self.mlp(x[0])
        return out
    
def load_model(file_path, model_name, model_architecture):
    _args = G4_args()
    _args.load(file_path, model_name)
    model = model_architecture(_args)
    model.load_state_dict(torch.load(file_path + model_name + '_weights.pth', weights_only=True))
    return model
    