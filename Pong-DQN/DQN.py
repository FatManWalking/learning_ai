import torch
import skimage.transform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from random import random, randint, sample


# Erster Versuch basierte auf visuellem Input. Einen State über Ball Position(x,y), Richtung und eigene Position konvergiert deutlich schneller
# Simpler = Better
# Habe Code für den Screencast in der Bild.py und der Vorverarbeitung (Downsampling) dennoch im Projekt gelassen
from Bild import Caster

resultion = (60, 40)

def preprocess(img, resultion):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resultion)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

#test = Caster()
#img = test.screencast()
#img = preprocess(img, resultion=resultion)
#print(img)

class PongPlayer(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self._create_weights()
        
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.fc(x)

        return x
    
def train(opt):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = PongPlayer()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    #TODO: Fix how and what to get in the state
    # Überall wo enc dran steht
    # Am besten das Pong Env immer auf das Modell warten lassen bis der nächste Step gegeben wird
    state = env.reset()
    
    model.to(device)
    state.to(device)

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        next_states = next_states.to(device)
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action, render=True)

        next_state = next_state.to(device)
        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            state = state.to(device)
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))


        state_batch = state_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/pong_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/pong".format(opt.saved_path))

def get_args():
    parser = argparse.ArgumentParser(
        """Deep Q Network to play Pong""")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--replay_memory_size", type=int, default=3000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--saved_path", type=str, default="models")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = get_args()
    train(opt)