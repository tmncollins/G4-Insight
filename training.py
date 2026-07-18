import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from dataset import G4_Dataset
from encoding import *
from model import G4_RNN, G4_RNN_context
from collections import *
import random
from time import time
import tqdm
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, recall_score, precision_score, roc_auc_score, f1_score, average_precision_score


torch.autocastptdtype = {
    'float32': torch.float32, 
    'bfloat16':     torch.bfloat16, 
    'float16': torch.float16
}

######## TRAINING ########

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mini_batches(class_a, class_b, batch_size, shuffle=True):
    if len(class_b) > len(class_a): return mini_batches(class_b, class_a, batch_size)
    b = batch_size // 2
    class_a_nums = [i for i in range(len(class_a))]
    class_b_nums = [i for i in range(len(class_b))]
    if shuffle:
        random.shuffle(class_a_nums)
        random.shuffle(class_b_nums)

    for i in range(len(class_a) // b):
        batch = []
        for j in range(b): 
            class_a.set_context(random.randint(50, 100))
            batch.append(class_a[class_a_nums[(b*i+j)]])
        for j in range(b): 
            class_b.set_context(random.randint(50, 100))
            batch.append(class_b[class_b_nums[(b*i+j)%len(class_b)]])

        random.shuffle(batch)
        yield batch


def train(rnn, training_data_true, training_data_false, all_training_data, validation_data, epochs=10, 
          batch_size=64, L=0.2, shuffle=True, tol=5, testing=False, save_best=False, device_name="cuda:0"):

    print("training :)")

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print("device", device)

    all_losses = [[], []]
    optimiser =  torch.optim.RAdam(rnn.parameters(), lr=L)
    scheduler = ReduceLROnPlateau(optimiser, "min")
    criterion = nn.BCEWithLogitsLoss()

    min_val_loss = float("inf")
    tol_cnt = 0

    best_rnn = deepcopy(rnn.state_dict())

    rnn.to(device)

    for e in range(1, epochs+1):
        rnn.zero_grad() # clear gradients
        rnn.train()
        current_loss = 0

        if testing and e % 10 == 0: 
            print(test(rnn, all_training_data))

        # create some minibatches
        # we cannot use dataloaders because each of our sequences is a different length
        t = time()

        batches = mini_batches(training_data_false, training_data_true, batch_size, shuffle)
        total_batches = max(len(training_data_true), len(training_data_false)) // (batch_size // 2)

        train_size = 0
        for batch in tqdm.tqdm(batches, desc="train loop", total=total_batches):
            optimiser.zero_grad()
            batch_loss = 0
            for i in batch:
                idx, upseq, pqs, downseq, target = i
                upseq = upseq.to(device); pqs = pqs.to(device); downseq = downseq.to(device); target = target.to(device)
                pred = rnn(upseq, pqs, downseq)
                loss = criterion(pred, target)
                batch_loss += loss

                del upseq; del pqs; del downseq

            batch_loss.backward()
            optimiser.step()

            current_loss += batch_loss.item() / len(batch)
            train_size += 1
        
        current_loss /= train_size

        rnn.eval()
        val_loss = 0

        with torch.no_grad():
            val_batches = list(range(len(validation_data)))
            random.shuffle(val_batches)
            val_batches = np.array_split(val_batches, len(val_batches) // batch_size)

            for batch in tqdm.tqdm(val_batches, desc="validation loop"):
                batch_loss = 0
                for i in batch:
                    idx, upseq, pqs, downseq, target = validation_data[i]
                    upseq = upseq.to(device); pqs = pqs.to(device); downseq = downseq.to(device); target = target.to(device)

                    pred = rnn(upseq, pqs, downseq)
                    loss = criterion(pred, target)
                    batch_loss += loss

                    del upseq; del pqs; del downseq

                val_loss += batch_loss.item() / len(batch)
            
            val_loss /= len(val_batches)

        scheduler.step(val_loss)
        all_losses[0].append(current_loss)
        all_losses[1].append(val_loss)        
        print(f"At epoch {e} out of {epochs}. Training loss: {current_loss}. Validation loss: {val_loss}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            tol_cnt = 0
            if save_best:
                best_rnn = deepcopy(rnn.state_dict())
        else:
            tol_cnt += 1
            if tol_cnt >= tol: break

    if save_best:
        rnn.load_state_dict(best_rnn)

    return all_losses

######## TESTING ########

def test(rnn, testing_data, batch_size=64, threshold=0.5, device_name="cuda:0"):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(device)

    correct = 0
    # create some minibatches
    # we cannot use dataloaders because each of our names is a different length
    test_batches = list(range(len(testing_data)))
    random.shuffle(test_batches)
    test_batches = np.array_split(test_batches, len(test_batches) // batch_size)

    pred_values = []
    pred_values_raw = []
    true_values = []

    rnn.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(test_batches, desc="testing loop"):
            for i in batch:
                idx, upseq, pqs, downseq, target = testing_data[i]
                upseq = upseq.to(device); pqs = pqs.to(device); downseq = downseq.to(device)

                try:
                    pred = rnn(upseq, pqs, downseq).detach().cpu().numpy()
                    pred = sigmoid(pred)
                    threshold_pred = pred[0] > threshold
                    correct += threshold_pred == target[0]
                    true_values.append(target[0])
                    pred_values.append(threshold_pred)
                    pred_values_raw.append(pred[0])
                except:
                    print(upseq, pqs, downseq)

                upseq = upseq.to("cpu"); pqs = pqs.to("cpu"); downseq = downseq.to("cpu")

    print(confusion_matrix(true_values, pred_values))
    print("Precision:", precision_score(true_values, pred_values))
    print("Recall:", recall_score(true_values, pred_values))
    print("F1:", f1_score(true_values, pred_values))
    print("Jaccard:", recall_score(true_values, pred_values))
    print("AUROC:", roc_auc_score(true_values, pred_values))
    print("AUPRC:", average_precision_score(true_values, pred_values))
    score = correct.item() / len(testing_data)
    print("Accuracy:", score)
    return score


######## LOAD DATA ########

def load_data(context, TRAIN_SIZE, ENCODING=None, VAL_SIZE=-1, TEST_SIZE=-1, DATASET=None, filename="/home/toby/RNN/all_pqs_seqs_flanked.txt"):
    f = open(filename).read().strip().split("\n")

    if DATASET == None:
        DATASET = G4_Dataset

    random.shuffle(f)
    if VAL_SIZE == -1:
        VAL_SIZE = max(500, TRAIN_SIZE // 100)
    if TEST_SIZE == -1:
        TEST_SIZE = max(1000, TRAIN_SIZE // 20)
    if not ENCODING:
        ENCODING = strand_to_int

    SAMPLE_SIZE = TRAIN_SIZE + VAL_SIZE

    f_train_yes = []
    f_train_no = []
    for line in f:
        if line[-1] == "1":
            if len(f_train_yes) < SAMPLE_SIZE:
                f_train_yes.append(line)
        elif len(f_train_no) < SAMPLE_SIZE:
            f_train_no.append(line)
        if len(f_train_yes) == SAMPLE_SIZE and len(f_train_no) == SAMPLE_SIZE:
            break

    print(len(f_train_yes), len(f_train_no), context)

    training_data_yes = DATASET(f_train_yes[VAL_SIZE:], ENCODING, context)
    training_data_no = DATASET(f_train_no[VAL_SIZE:], ENCODING, context)
    all_training_data = DATASET(f_train_yes[VAL_SIZE:] + f_train_no[VAL_SIZE:], ENCODING, context)
    validation_data = DATASET(f_train_yes[:VAL_SIZE] + f_train_no[:VAL_SIZE], ENCODING, context, debug=True)
    test_data = DATASET(f[SAMPLE_SIZE:TRAIN_SIZE+TEST_SIZE], ENCODING, context)

    return (training_data_yes, training_data_no, all_training_data), validation_data, test_data

def load_data_by_chromosome(context, train_chrom, val_chrom, test_chrom, TRAIN_SIZE, VAL_SIZE=-1, TEST_SIZE=-1, 
                            ENCODING=None, DATASET=None, filename="/home/toby/RNN/all_pqs_seqs_flanked.txt"):
    f = open(filename).read().strip().split("\n")

    if DATASET == None:
        DATASET = G4_Dataset

    random.shuffle(f)
    if VAL_SIZE == -1:
        VAL_SIZE = max(500, TRAIN_SIZE // 100)
    if TEST_SIZE == -1:
        TEST_SIZE = max(1000, TRAIN_SIZE // 20)
    if not ENCODING:
        ENCODING = strand_to_int

    f_train_yes = []
    f_train_no = []
    f_val_yes = []
    f_val_no = []
    f_test = []
    for _line in f:
        line = _line.split(",")
        if line[-2] in train_chrom:
            if line[-1] == "1":
                if len(f_train_yes) < TRAIN_SIZE: f_train_yes.append(_line)
            elif len(f_train_no) < TRAIN_SIZE: f_train_no.append(_line)
        elif line[-2] in val_chrom:
            if line[-1] == "1":
                if len(f_val_yes) < VAL_SIZE: f_val_yes.append(_line)
            elif len(f_val_no) < VAL_SIZE: f_val_no.append(_line)
        elif line[-2] in test_chrom:
            f_test.append(_line)

    print("train data: ", len(f_train_yes), len(f_train_no))
    print("val data: ", len(f_val_yes), len(f_val_no))
    print("test data: ", len(f_test))
    training_data_yes = DATASET(f_train_yes, ENCODING, context)
    training_data_no = DATASET(f_train_no, ENCODING, context)
    all_training_data = DATASET(f_train_yes + f_train_no, ENCODING, context)
    validation_data = DATASET(f_val_yes + f_val_no, ENCODING, context)
    test_data = DATASET(f_test[:TEST_SIZE], ENCODING, context)

    return (training_data_yes, training_data_no, all_training_data), validation_data, test_data
