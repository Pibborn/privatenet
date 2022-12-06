from utils import load_data, split_parties, split_x_y, get_n_features, encode_splits, combine_splits
from network import Net
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
import torch
from timeit import default_timer as timer
import sys

NUM_PARTIES = int(sys.argv[1])
DATA_LOC = ['input_data_bnetflix.in']
LABELS_LOC = ['target_values_bnetflix.in']
RANDOM_STATE = 999
BATCH_SIZE = 32
N_EPOCHS = 1
LR = 1e-5

@mpc.run_multiprocess(world_size=NUM_PARTIES)
def train(data_splits, enc_model):
    data_splits = encode_splits(data_splits)
    x_combined, y_combined = combine_splits(data_splits)
    num_batches = x_combined.size(0) // BATCH_SIZE
    enc_model.train()
    loss = crypten.nn.MSELoss()
    rank = comm.get().get_rank()
    print('Num batches per epoch: {}'.format(num_batches))
    for i in range(N_EPOCHS):
        crypten.print(f"Epoch {i} in progress:")

        for i, batch in enumerate(range(num_batches)):
            start_batch = timer()
            # define the start and end of the training mini-batch
            start, end = batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE

            # construct CrypTensors out of training examples / labels
            x_train = x_combined[start:end]
            y_train = y_combined[start:end]

            # perform forward pass:
            output = model(x_train)

            loss_value = loss(output, y_train)

            # set gradients to "zero"
            enc_model.zero_grad()

            # perform backward pass:
            loss_value.backward()

            # update parameters
            enc_model.update_parameters(LR)

            # Print progress every batch:
            #batch_loss = loss_value.get_plain_text()
            end_batch = timer()
            if i % 10 == 0 or i == 0:
                print('Batch time: {}'.format(end_batch-start_batch))



if __name__ == '__main__':
    crypten.init()
    for data_loc_single, labels_loc_single in zip(DATA_LOC, LABELS_LOC):
        df = load_data(data_loc_single, labels_loc_single)
        splits = split_parties(df, n_parties=NUM_PARTIES, random_state=RANDOM_STATE)
        n_features = get_n_features(split_x_y(splits[0])[0])
        nn = Net(n_features)
        crypten.common.serial.register_safe_class(Net)
        #torch.set_num_threads(1)
        dummy_input = torch.empty(1, n_features)
        model = crypten.nn.from_pytorch(nn, dummy_input)
        model.encrypt()
        print('data {} parties {}'.format(data_loc_single, NUM_PARTIES))
        start = timer()
        train(splits, model)
        end = timer()
        print('Epoch time: {}'.format(end-start))
