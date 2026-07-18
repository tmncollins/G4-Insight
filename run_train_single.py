from training import train, test, load_data, load_data_by_chromosome
from model import G4_args, G4_RNN, G4_RNN_ReLU
from collections import namedtuple
import matplotlib.pyplot as plt
from dataset import *
import torch

context = 150

training_args = namedtuple('training_args', ['train_size', 'val_size', 'test_size','epochs', 'tol', 'batch_size', 'L'])
training_args.train_size = 1000000
training_args.val_size = 10000
training_args.test_size = 1000000
training_args.epochs = 10
training_args.tol = 3
training_args.L = 0.0005
training_args.batch_size = 256

_args = G4_args()
_args.vocab_size = 5
_args.cnn_channels_in = 128
_args.cnn_channels_out = 128
_args.cnn_kernel_size = 16
_args.rnn_layer_size = 128
_args.rnn_layer_n = 5
_args.rnn_dropout = 0.10
_args.mlp_dropout = 0.20
_args.mlp_layer_size = 256
_args.rnn_bidirectional = True
_args.pool_size = 3
_args.context = context
_args.L = training_args.L

model = G4_RNN_ReLU(_args)

device="cuda:0"
model_name = "model_full_data_context_150_relu"

print(f"Running {model_name} on {device}")

# --- Prepare Data ---
train_chrom = ["chr" + str(i) for i in range(3, 23)] + ["chrX", "chrY", "chrM"]
val_chrom = ["chr2"]
test_chrom = ["chr1"]
#train_data, val_data, test_data = load_data_by_chromosome(context, train_chrom, val_chrom, test_chrom, training_args.train_size, VAL_SIZE=training_args.val_size,
#                                            TEST_SIZE=training_args.test_size, DATASET = G4_Dataset)
train_data, val_data, test_data = load_data(context, training_args.train_size, VAL_SIZE=training_args.val_size,
                                            TEST_SIZE=training_args.test_size, DATASET = G4_Dataset,
                                            filename='Data/G4_comparison.txt')
train_data_yes, train_data_no, train_data_all = train_data

# --- Training ---
_args.save("Models/", model_name)
loss = train(model, train_data_yes, train_data_no, train_data_all, val_data, epochs=training_args.epochs, 
             batch_size=training_args.batch_size, L=training_args.L, tol=training_args.tol, device_name=device,
             save_best=True)
torch.save(model.state_dict(), "Models/" + model_name + "_weights.pth")

# --- Evaluate ---
print(test(model, val_data, device_name=device))
print(test(model, test_data, device_name=device))

plt.plot(loss[0], label="Training Loss")
plt.plot(loss[1], label="Validation Loss")
plt.legend()
plt.savefig("RNN/Models/Loss/" + model_name + "_loss.png")
