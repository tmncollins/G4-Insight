import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import G4_RNN, G4_args
from training import train, test, load_data, load_data_by_chromosome
from collections import namedtuple
from dataset import G4_Dataset

import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig

def train_and_evaluate(model_args, train_data, val_data, test_data):
    global training_args

    train_data_yes, train_data_no, train_data_all = train_data

    # model
    _args = G4_args()
    _args.vocab_size = model_args["vocab_size"]
    _args.cnn_channels_in = model_args["cnn_channels_in"]
    _args.cnn_channels_out = model_args["cnn_channels_out"]
    _args.cnn_kernel_size = model_args["cnn_kernel_size"]
    _args.rnn_layer_size = model_args["rnn_layer_size"]
    _args.rnn_layer_n = model_args["rnn_layer_n"]
    _args.rnn_dropout = model_args["rnn_dropout"]
    _args.mlp_layer_size = model_args["mlp_layer_size"]
    _args.mlp_dropout = model_args["mlp_dropout"]
    _args.rnn_bidirectional = model_args["bidirectional"]
#    _args.pool_size = model_args["pool_size"]
    model = G4_RNN(_args)

    # --- Training ---
    loss = train(model, train_data_yes, train_data_no, train_data_all, val_data, epochs=training_args.epochs, 
                batch_size=model_args["batch_size"], L=model_args["L"], tol=training_args.tol, save_best=True,
                device_name=dev)
    # --- Evaluate ---
    return test(model, val_data, device_name=dev)


training_args = namedtuple('training_args', ['train_size', 'val_size', 'test_size','epochs', 'tol'])
training_args.train_size = 700000
training_args.val_size = 10000
training_args.test_size = 900000
training_args.epochs = 3
training_args.tol = 3
context = 150

print("finetuning")

dev = 'cuda:0'

client = Client()

parameters = [
    ChoiceParameterConfig(name="L", parameter_type="float", values=[0.0005]),
    ChoiceParameterConfig(name="bidirectional", parameter_type="bool", values=[True, False]),
#    RangeParameterConfig(name="context", parameter_type="int", bounds=(0,75)),
    ChoiceParameterConfig(name="vocab_size", parameter_type="int", values=[5]),
    RangeParameterConfig(name="cnn_channels_in", parameter_type="int", bounds=(8, 256)),
    RangeParameterConfig(name="cnn_channels_out", parameter_type="int", bounds=(8, 256)),
    RangeParameterConfig(name="cnn_kernel_size", parameter_type="int", bounds=(2, 32)),
    RangeParameterConfig(name="rnn_layer_size", parameter_type="int", bounds=(32, 256)),
    RangeParameterConfig(name="rnn_layer_n", parameter_type="int", bounds=(2, 8)),
    RangeParameterConfig(name="rnn_dropout", parameter_type="float", bounds=(0.0, 0.25)),
    RangeParameterConfig(name="mlp_dropout", parameter_type="float", bounds=(0.0, 0.25)),
    RangeParameterConfig(name="mlp_layer_size", parameter_type="int", bounds=(32, 512)),
#    RangeParameterConfig(name="pool_size", parameter_type="int", bounds=(1,5)),
    ChoiceParameterConfig(name="batch_size", parameter_type="int", values=[256]),
]

client.configure_experiment(parameters=parameters)
objective = "score"
client.configure_optimization(objective=objective)

trials_cnt = 50

# --- Prepare Data ---
train_chrom = ["chr" + str(i) for i in range(3, 23)] + ["chrX", "chrY", "chrM"]
val_chrom = ["chr2"]
test_chrom = ["chr1"]
train_data, val_data, test_data = load_data_by_chromosome(context, train_chrom, val_chrom, test_chrom, training_args.train_size, VAL_SIZE=training_args.val_size,
                                            TEST_SIZE=training_args.test_size, DATASET = G4_Dataset)
#train_data, val_data, test_data = load_data(context, training_args.train_size, VAL_SIZE=training_args.val_size,
#                                            TEST_SIZE=training_args.test_size, DATASET = G4_Dataset)
train_data_yes, train_data_no, train_data_all = train_data

CNT = trials_cnt
for _ in range(CNT):
    trials = client.get_next_trials(max_trials=3)

    for idx, param in trials.items():
        print(param)
        trials_cnt -= 1

        score = 0
        try:
            score = train_and_evaluate(param, train_data, val_data, test_data)
        except Exception as e:
            print(e)

        print(score)
        raw_data = {objective: score}

        client.complete_trial(trial_index=idx, raw_data=raw_data)
        if trials_cnt <= 0: break

best_parameters, prediction, index, name = client.get_best_parameterization()
print("Best Parameters:", best_parameters)
print("Prediction (mean, variance):", prediction)

client.save_to_json_file("bo_snapshot.json")