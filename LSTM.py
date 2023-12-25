import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "game_tensorsd1.pt"
loaded_data = torch.load(file_path)

all_game_tensors = loaded_data


# try bucking the data instead?
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Using output of the last round

        return out, (hn, cn)


# experiment with this
hidden_size = 128
# 244 features - 5 at the end which are the labels
input_size = 239
output_size = 5
num_layers = 2
learning_rate = 0.001

model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 1000  # Specify the number of epochs

for epoch in range(num_epochs):
    for game_tensor in all_game_tensors:
        # Initialize hidden and cell states for each game
        hn, cn = None, None

        # Process each round in the game sequentially
        for round_tensor in game_tensor:
            # round_tensor = round_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 244)
            input_features = round_tensor[:-5].unsqueeze(0).unsqueeze(0)
            # Convert one-hot encoded target to class indices
            target_one_hot = round_tensor[-5:].unsqueeze(0)
            target_indices = torch.argmax(target_one_hot, dim=1).to(device)

            # Pass the round through the LSTM model along with the previous states
            round_output, (hn, cn) = model(input_features, hn, cn)

            # Calculate the loss
            loss = criterion(round_output, target_indices)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
torch.save(model.state_dict(), "LSTM.pth")
