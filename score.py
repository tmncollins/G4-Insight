from model import *
from training import *
from torchsummary import summary
from sys import exit
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import tqdm
import random

random.seed(42)

model_name = "model_full_data_context_150_relu"

_args = G4_args()

_args.load(model_name)
_args.context = 150
device="cuda:0"

model = G4_RNN_ReLU(_args)
model.load_state_dict(torch.load('/home/toby/RNN/Models/' + model_name + '_weights.pth', weights_only=True))
###########################################
train_chrom = ["chr" + str(i) for i in range(4, 23)] + ["chrX", "chrY", "chrM"]
val_chrom = ["chr3"]
test_chrom = ["chr1", "chr2"]
train_data, val_data, test_data = load_data_by_chromosome(_args.context, train_chrom, val_chrom, test_chrom, 
                                        50, VAL_SIZE=2000, TEST_SIZE=1000000, DATASET = G4_Dataset,
#                                        filename='/home/toby/G4_comparison.txt')
                                        filename='/home/toby/RNN/all_pqs_seqs_flanked.txt')

#train_data, val_data, test_data = load_data(_args.context,
#                                        50, VAL_SIZE=2000, TEST_SIZE=1_000_000, DATASET = G4_Dataset,
#                                        filename='/home/toby/RNN/all_pqs_seqs_flanked.txt')

train_data_yes, train_data_no, train_data_all = train_data
print(model)

t = 0.995
train_data_yes, train_data_no, train_data_all = train_data
model.to(device)
print(test(model, test_data, device_name=device, threshold=t))

MIN_T = 2
MAX_T = 1
t = MIN_T
#while t <= MAX_T:
for t in [0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.93,0.95,0.96,0.97,0.98,0.99]:
    #print(test(model, train_data_all, device_name=device, threshold=0.5))
    #print(test(model, val_data, device_name=device, threshold=t))
    print("t", t)
    print(test(model, test_data, device_name=device, threshold=t))
#    t += 0.01
