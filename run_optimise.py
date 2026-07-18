import torch
import re
import numpy as np
import random
from model import G4_args, G4_RNN, load_model
from encoding import strand_to_int as encode
from optimise import *

device= "cuda:0"
model_name = "model_full_data_context_20"

_args = G4_args()
_args.load(model_name)
model = G4_RNN(_args)
model.load_state_dict(torch.load('/home/toby/RNN/Models/' + model_name + '_weights.pth', map_location=torch.device("cpu"), weights_only=True))
model.to(device)
print('loaded model', model_name, 'to', device)

print(predict_logit(model, "AGGGTGGGGAGGGTGGGG", device))

for i in range(5):
    print(optimise_pqs_core(model, device, "AGGGTGGGGAGGGTGGGG", 19, 400, 20, verbose=False, method="max"))