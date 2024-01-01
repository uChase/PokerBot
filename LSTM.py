import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path1 = "data\IlxxxlI\d1train.pt"
file_path2 = "data\IlxxxlI\d2train.pt"
file_path3 = "data\IlxxxlI\d3train.pt"

loaded_data1 = torch.load(file_path1)
loaded_data2 = torch.load(file_path2)
loaded_data3 = torch.load(file_path3)

all_game_tensors = loaded_data1 + loaded_data2 + loaded_data3


# try bucking the data instead?
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)  # Add dropout layer


    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out)  # Apply dropout

        out = self.fc(out[:, -1, :])  # Using output of the last round

        return out, (hn, cn)


# experiment with this
hidden_size = 256
# 244 features - 5 at the end which are the labels
# input_size = 239
# output_size = 5
# new input size 230, since 234 features - 4 at the end which are the labels
input_size = 230
output_size = 4
num_layers = 3
learning_rate = 0.001

model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 50  # Specify the number of epochs
# Train the model
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    total_rounds = 0
    for game_tensor in all_game_tensors:
        # Initialize hidden and cell states for each game
        hn, cn = None, None

        # Process each round in the game sequentially
        for round_tensor in game_tensor:
            input_features = round_tensor[:-4].unsqueeze(0).unsqueeze(0).to(device)
            target_one_hot = round_tensor[-4:].unsqueeze(0)
            target_indices = torch.argmax(target_one_hot, dim=1).to(device)

            # Pass the round through the LSTM model along with the previous states
            round_output, (hn, cn) = model(input_features, (hn.detach() if hn is not None else None,
                                                           cn.detach() if cn is not None else None))

            # Calculate the loss
            loss = criterion(round_output, target_indices)
            total_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Detach the hidden and cell states from the graph
            hn = hn.detach()
            cn = cn.detach()
            total_rounds += 1
    average_loss = total_loss / total_rounds if total_rounds > 0 else 0
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")
torch.save(model.state_dict(), "LSTM3.pth")


loaded_test1 = torch.load("data\IlxxxlI\\t1.pt")
loaded_test2 = torch.load("data\IlxxxlI\\t2.pt")
loaded_test3 = torch.load("data\IlxxxlI\\t3.pt")
all_test_tensors = loaded_test1 + loaded_test2 + loaded_test3
model.eval()  # Set the model to evaluation mode

total_correct = 0
total_samples = 0



with torch.no_grad():
    for test_tensor in all_test_tensors:
        hn, cn = None, None

        for round_tensor in test_tensor:
            input_features = round_tensor[:-4].unsqueeze(0).unsqueeze(0).to(device)
            target_one_hot = round_tensor[-4:].unsqueeze(0)
            target_indices = torch.argmax(target_one_hot, dim=1).to(device)

            round_output, (hn, cn) = model(input_features, (hn.detach() if hn is not None else None,
                                                           cn.detach() if cn is not None else None))

            _, predicted_indices = torch.max(round_output, dim=1)
            total_correct += (predicted_indices == target_indices).sum().item()
            total_samples += target_indices.size(0)

accuracy = total_correct / total_samples if total_samples > 0 else 0
print(f"Testing Accuracy: {accuracy:.4f}")


