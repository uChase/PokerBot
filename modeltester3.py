import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loaded_test1 = torch.load("data\IlxxxlI\\t1.pt")
loaded_test2 = torch.load("data\IlxxxlI\\t2.pt")
loaded_test3 = torch.load("data\IlxxxlI\\t3.pt")
all_test_tensors = loaded_test1 + loaded_test2 + loaded_test3


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
# new input size 230, since 234 features - 4 at the end which are the labels
input_size = 230
output_size = 4
num_layers = 3
learning_rate = 0.001

model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('./models/LSTM3.pth'))

model.eval()  # Set the model to evaluation mode

total_correct = 0
total_samples = 0
fold_false = 0
check_false = 0
call_false = 0
raise_false = 0
fold_correct = 0
check_correct = 0
call_correct = 0
raise_correct = 0


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

            if predicted_indices != target_indices:
                if target_indices.item() == 0:
                    fold_false += 1
                elif target_indices.item() == 1:
                    check_false += 1
                elif target_indices.item() == 2:
                    call_false += 1
                elif target_indices.item() == 3:
                    raise_false += 1
            else:
                if target_indices.item() == 0:
                    fold_correct += 1
                elif target_indices.item() == 1:
                    check_correct += 1
                elif target_indices.item() == 2:
                    call_correct += 1
                elif target_indices.item() == 3:
                    raise_correct += 1
            total_samples += target_indices.size(0)
            
accuracy = total_correct / total_samples if total_samples > 0 else 0
print(f"Testing Accuracy: {accuracy:.4f}")
print("Fold False: ", fold_false)
print("Check False: ", check_false)
print("Call False: ", call_false)
print("Raise False: ", raise_false)

print("Fold Correct: ", fold_correct)
print("Check Correct: ", check_correct)
print("Call Correct: ", call_correct)
print("Raise Correct: ", raise_correct)

print("percentage of folds correct" , (fold_correct/(fold_correct + fold_false))*100)
print("percentage of checks correct" , (check_correct/(check_correct + check_false))*100)
print("percentage of calls correct" , (call_correct/(call_correct + call_false))*100)
print("percentage of raises correct" , (raise_correct/(raise_correct + raise_false))*100)
