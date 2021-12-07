"""
Author: Said Kaloev
Exercise 5: Machine Learning Project
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from architectures import SimpleCNN
from torch.utils.tensorboard import SummaryWriter
import tqdm
import dill as pkl
from datasets import ImageData

#Hyperparameters to change
l_rate = 1.0001e-3
weight_decay = 1e-5
n_updates = 5e4
batch_size = 1
num_workers = 4
n_hidden_layers = 3
n_in_channels = 2
n_kernels = 32
kernel_size = 7


def separate_sets():
    """This function is needed to split the data into different sets"""
    data_set = ImageData()
    set = data_set
    trainingset = torch.utils.data.Subset(set, indices=np.arange(int(len(set) * (3 / 5))))
    validationset = torch.utils.data.Subset(set, indices=np.arange(int(len(set) * (3 / 5)),
                                                                   int(len(set) * (4 / 5))))
    testset = torch.utils.data.Subset(set, indices=np.arange(int(len(set) * (4 / 5)),
                                                             len(set)))
    return trainingset, validationset, testset


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`"""
    # Define a loss (mse loss)
    mse = torch.nn.MSELoss()
    # We will accumulate the mean loss in variable `loss`
    loss = torch.tensor(0., device=device)
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0, colour="green"):
            # Get a sample and move inputs and targets to device
            image_target, inputs, known = data
            inputs = inputs.float()
            known = known.float()
            image_target = image_target.float()
            inputs = inputs.to(device)
            known = known.to(device)
            image_target = image_target.to(device)
            inputs = inputs.reshape(-1, 1, 90, 90)
            known = known.reshape(-1, 1, 90, 90)
            image_target = image_target.reshape(-1, 1, 90, 90)
            stacked_elements = torch.cat((inputs, known), dim=1)
            stacked_elements = stacked_elements.to(device)
            outputs = model(stacked_elements)

            torch.clamp(outputs, 0, 255)

            loss += (torch.stack([mse(output, target) for output, target in zip(outputs, image_target)]).sum()
                     / len(dataloader.dataset))
    return loss

trainingset, validationset, testset = separate_sets()

train_loader = torch.utils.data.DataLoader(trainingset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
validation_loader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)

# Create Network
net = SimpleCNN(n_hidden_layers=n_hidden_layers,
                n_in_channels=n_in_channels,
                n_kernels=n_kernels,
                kernel_size=kernel_size)
device = torch.device('cpu')
net.to(device)

results_path = 'model'

# Get mse loss function
mse = torch.nn.MSELoss()

# Get adam optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=l_rate, weight_decay=weight_decay)

writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))

print_stats_at = 1e2  # print status to tensorboard every x updates
plot_at = 1e4  # plot every x updates
validate_at = 5e3  # evaluate model on validation set and check for new best model every x updates
update = 0  # current update counter
best_validation_loss = np.inf  # best validation loss so far
update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

# Save initial model as "best" model (will be overwritten later)
torch.save(net, os.path.join(results_path, 'best_model.pt'))
    
# Train until n_updates update have been reached

while update < n_updates:
    for data in train_loader:
        # Get next samples in `trainloader_augmented`
        image_target, inputs, known = data
        inputs = inputs.float()
        known = known.float()
        image_target = image_target.float()

        inputs = inputs.to(device)
        inputs = inputs.reshape(-1, 1, 90, 90)
        known = known.reshape(-1, 1, 90, 90)
        image_target = image_target.reshape(-1, 1, 90, 90)
        known = known.to(device)
        image_target = image_target.to(device)

        stacked = torch.cat((inputs, known), dim=1)
        stacked = stacked.to(device)
        # Reset gradients
        optimizer.zero_grad()

        # Get outputs for network
        outputs = net(stacked)

        # Calculate loss, do backward pass, and update weights
        loss = mse(outputs, image_target)
        loss.backward()
        optimizer.step()

        # Print current status and score
        if update % print_stats_at == 0 and update > 0:
            writer.add_scalar(tag="training/loss",
                              scalar_value=loss.cpu(),
                              global_step=update)

        # Evaluate model on validation set
        if update % validate_at == 0 and update > 0:
            val_loss = evaluate_model(net, dataloader=validation_loader, device=device)
            writer.add_scalar(tag="validation/loss", scalar_value=val_loss.to(device), global_step=update)
            # Add weights as arrays to tensorboard
            for i, param in enumerate(net.parameters()):
                writer.add_histogram(tag=f'validation/param_{i}', values=param.to(device),
                                     global_step=update)
            # Add gradients as arrays to tensorboard
            for i, param in enumerate(net.parameters()):
                writer.add_histogram(tag=f'validation/gradients_{i}',
                                     values=param.grad.to(device),
                                     global_step=update)
            # Save best model for early stopping
            if best_validation_loss > val_loss:
                best_validation_loss = val_loss
                torch.save(net, os.path.join(results_path, 'best_model.pt'))

        update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
        update_progess_bar.update()

        # Increment update counter, exit if maximum number of updates is reached
        update += 1
        if update >= n_updates:
            break

update_progess_bar.close()
print('We just finished Training! \n')
print("-------------------------------------------------------")

# Load best model and compute score on test set
print(f"Start computing scores for the best model.......................")
print("-----------------------------------------------------------------")
net = torch.load(os.path.join(results_path, 'best_model.pt'))
test_loss = evaluate_model(net, dataloader=test_loader, device=device)
val_loss = evaluate_model(net, dataloader=validation_loader, device=device)
train_loss = evaluate_model(net, dataloader=train_loader, device=device)

print(f"The scores are..........")
print("-----------------------------------------------------------------")
print(f"Loss of Testset is: {test_loss}")
print(f"Loss of Validationset is: {val_loss}")
print(f"Loss of Trainingset is: {train_loss}")

# Write result to file
with open(os.path.join(results_path, 'results.txt'), 'w') as fh:
    print(f"Scores:", file=fh)
    print(f"test loss: {test_loss}", file=fh)
    print(f"validation loss: {val_loss}", file=fh)
    print(f"training loss: {train_loss}", file=fh)

test_set = pd.read_pickle(r'challenge_testset/testset.pkl')
complete_mess = []
for input, known in zip(test_set['input_arrays'], test_set['known_arrays']):
    complete_mess.append((input, known))

result = np.array(complete_mess)
special_loader = torch.utils.data.DataLoader(result, batch_size=1, shuffle=False, num_workers=num_workers)

# This is needed to save outputs in array list, and save this array list as a pickle_file
print("Start processing results into array............")
some_arrays = []
for data in special_loader:
    inputs, known = data[0]
    inputs = inputs.float()
    known = known.float()
    inputs = inputs.to(device)
    known = known.to(device)
    inputs = inputs.reshape(-1, 1, 90, 90)
    known = known.reshape(-1, 1, 90, 90)

    stacked_elements = torch.cat((inputs, known), dim=1)
    stacked_elements = stacked_elements.to(device)

    outputs = net(stacked_elements)
    outputs = outputs.cpu().detach().numpy()

    image = outputs[0][0]
    inputs = inputs.cpu().detach().numpy()
    known = known.cpu().detach().numpy()
    image = np.where(known, 0, image)
    image = image[image != 0]
    image = image.flatten()
    image = image.astype('uint8')

    some_arrays.append(image)
print("-------------------------------------------")
print("Process of creating the outputs has finished.")
with open(f'outputs.pkl', 'wb') as fh:
    pkl.dump(some_arrays, file=fh)

