import A3C
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import pokergame
import pokerutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# INIT LSTM OPPONENTS
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


model = LSTM(239, 256, 2, 5).to(device)
model.load_state_dict(
    torch.load("./models/LSTM2.pth", map_location=torch.device(device))
)


input_dim = 215
output_dim = 3
num_layers = 2
gamma = 0.99

global_model_class = A3C.GlobalModel(input_dim, output_dim, num_layers)
global_model = global_model_class.get_model()
global_model.share_memory()


def workerFunc(global_model, input_dim, output_dim, num_layers, gamma, thread_num):
    worker = A3C.Worker(global_model, gamma, input_dim, output_dim, num_layers)
    playerList = [pokergame.Player(str(i + 1), 1000) for i in range(9)]
    totalWealth = 9000
    epochs = 25000
    for epoch in range(epochs):
        players = []
        if epoch % 50 == 0:
            print("Thread:", thread_num, "Epoch:", epoch)
        if epoch % 15 == 0:
            resetMoney = True
        if resetMoney:
            for player in playerList:
                player.set_money(1000)
        game = pokergame.PokerGame()
        for player in playerList:
            game.add_player(player)
        gameOver = False
        enough = game.check_enough()
        if not enough:
            continue
        currplayer, cardStatus, community_cards, pot, roundIn = game.startGame()
        viewable_cards = []
        prob_value_pairs = []
        while not gameOver:
            (
                gameStatus,
                currplayer,
                players,
                pot,
                cardStatus,
                roundIn,
            ) = game.check_round_redo()
            if gameStatus == "end":
                gameOver = True
                continue
            if currplayer["status"] == "out":
                (
                    gameStatus,
                    currplayer,
                    players,
                    pot,
                    cardStatus,
                    roundIn,
                ) = game.next_turn("null")
                continue
            if gameStatus == "redo":
                if cardStatus == 1:
                    viewable_cards = community_cards[:3]
                elif cardStatus == 2:
                    viewable_cards = community_cards[:4]
                elif cardStatus == 3:
                    viewable_cards = community_cards[:5]
            lstmInput = pokerutils.convert_round_to_tensor_LSTM(
                game.players, viewable_cards, pot, roundIn, currplayer, game.totalWealth
            )
            lstmOut = []
            lstmIndex = 0
            move = None
            if currplayer["player"].get_name() != "1":
                with torch.no_grad():
                    hn = currplayer["player"].get_hn()
                    cn = currplayer["player"].get_cn()
                    lstmInput = lstmInput.unsqueeze(0).to(device)
                    round_output, (hn, cn) = model(
                        lstmInput,
                        (
                            hn.detach() if hn is not None else None,
                            cn.detach() if cn is not None else None,
                        ),
                    )
                    currplayer["player"].set_hn(hn)
                    currplayer["player"].set_cn(cn)
                    softmax = nn.Softmax(dim=1)
                    softmax_output = softmax(round_output)
                    lstmOut = softmax_output.squeeze().tolist()
                    lstmIndex = torch.argmax(softmax_output).item()
                if lstmIndex == 0:
                    move = "fold"
                elif lstmIndex == 1 or lstmIndex == 2:
                    move = "c"
                elif lstmIndex == 3 or lstmIndex == 4:
                    move = "raise"
            else:
                ahn = currplayer["player"].get_hn()
                acn = currplayer["player"].get_cn()
                chn = currplayer["player"].get_hn2()
                ccn = currplayer["player"].get_cn2()
                stateTensor = pokerutils.convert_round_to_tesnor_DQN(
                    game.players,
                    viewable_cards,
                    pot,
                    roundIn,
                    currplayer,
                    lstmOut,
                    game.totalWealth,
                )
                stateTensor = stateTensor.to(device)
                action, policy, (ahn, acn), (chn, ccn), value = worker.choose_action(
                    stateTensor, ahn, acn, chn, ccn
                )
                currplayer["player"].set_hn(ahn)
                currplayer["player"].set_cn(acn)
                currplayer["player"].set_hn2(chn)
                currplayer["player"].set_cn2(ccn)
                prob_value_pairs.append(
                    policy, value, currplayer["player"].get_money(), pot, roundIn
                )
                if action == 0:
                    move = "fold"
                elif action == 1:
                    move = "c"
                elif action == 2:
                    move = "raise"
                else:
                    print("ERROR")
            gameStatus, currplayer, players, pot, cardStatus, roundIn = game.next_turn(
                move
            )

        print("game finished " + str(epoch), "thread:", thread_num)
        for player in players:
            player["player"].reset_hn()
            player["player"].reset_cn()
            if player["player"].get_name() == "1":
                endReward = player["Q"]
                endResult = player["result"]
                for i in range(len(prob_value_pairs)):
                    policy, value, bankroll, pot, roundIn = prob_value_pairs[i]
                    done = i == len(prob_value_pairs) - 1
                    reward = 0
                    if done:
                        reward = endReward
                    else:
                        reward = pokerutils.calculate_reward(
                            bankroll, action, endResult, pot, roundIn, totalWealth
                        )
                    reward *= 10**6
                    worker.update_buffer(policy, reward, value)
                worker.train()


if __name__ == "__main__":
    num_processes = 4  # Or any other number depending on your system's capabilities
    processes = []

    for thread_num in range(num_processes):
        p = mp.Process(
            target=workerFunc,
            args=(global_model, input_dim, output_dim, num_layers, gamma, thread_num),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    global_model_class.save_model("./models/A3C.pth")
