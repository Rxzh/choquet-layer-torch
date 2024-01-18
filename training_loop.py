import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from choquet import Choquet

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PATHOUT=...

BATCH_SIZE = int(...)
EPOCHS = int(...)

batch_size = BATCH_SIZE
epochs = EPOCHS
learning_rate=...
CHANNELS=...
NLAYERS=int(...)
NFILTERS=int(...)
KSIZE=int(...)
SHRINK=int(...)
SUBSPACE=int(...)
PATIENCE_ES=int(...)
PATIENCE_RP=int(...)
pathoutput=str(...)
output_dir_root =pathoutput

listIm=...
listY=...
# Convert your data to PyTorch tensors
tensor_x = torch.Tensor(listIm).to(device) # Assuming listIm is a list of your images
tensor_y = torch.Tensor(listY).to(device) # Assuming listY is your labels



# Create dataset and dataloaders
train_dataset = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# For validation data
tensor_x_val = torch.Tensor(listImVal).to(device) # Assuming listImVal is a list of your validation images
tensor_y_val = torch.Tensor(listYVal).to(device) # Assuming listYVal is your validation labels
val_dataset = TensorDataset(tensor_x_val, tensor_y_val)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

# Model, loss function and optimizer
model = Choquet(num_layers=NLAYERS, num_filters=NFILTERS, ksize=KSIZE, shrink=SHRINK, subspace=SUBSPACE, channels=CHANNELS)
model.to(device)
criterion = torch.nn.L1Loss() # MAE loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Reduce LR on plateau
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE_RP, factor=0.1, min_lr=1e-6)

# Early stopping logic
early_stopping_patience = PATIENCE_ES
early_stopping_counter = 0
best_loss = float("inf")

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_loss:
        best_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

    # Logging (you may want to add more detailed logging)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

# Save the model state if necessary
torch.save(model.state_dict(), 'path_to_save_model.pth')
