import random 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

LEARN_R = 0.0001
EPOCHS = 200
TEST_SAMPLE = 1000

def get_train_set(ammount: int=99999):
    df = pd.read_csv("data.csv", sep=";")
    
    data = df.drop(columns=["Target"])
    label = df["Target"]
    
    data_ts = []
    label_ts = []
    
    i = 0
    for d, l in zip(data.iterrows(), label):
        if i == ammount:
            break
        ds = [float(e) for e in d[1].values]
        data_ts.append(ds)
        o = [0.0, 0.0, 1.0]
        if l == "Graduate":
            o = [1.0, 0.0, 0.0]
        elif l == "Enroled":
            o = [0.0, 1.0, 0.0]
        label_ts.append(o)
        i += 1
    return torch.tensor(data_ts), torch.tensor(label_ts)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(36, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 3),
        )
    def forward(self, x):
        return self.layer_stack(x)

model = Network()


inputs, labels = get_train_set() 
print(f"{len(inputs)} examples \t Learning Rate: {LEARN_R}\t Epochs: {EPOCHS}")
print(f"example input: {inputs[0]}: {labels[0]}")

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

loss_fn = nn.BCEWithLogitsLoss()
opt = optim.SGD(model.parameters(), lr=LEARN_R)

model.train()
for i in range(EPOCHS):
    avg_loss = 0
    for b, (x, y) in enumerate(dataloader):
        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        opt.step()
        opt.zero_grad()
        avg_loss += loss.item()
    if i % 10 == 0:
        print(f"{i}: {avg_loss / 128}")


def get_idx(n, t):
    for i, e in enumerate(t):
        if e == n:
            return i
    return -1
def max_t(t):
    m = torch.inf * -1
    for e in t:
        if e >= m:
            m = e
    return m

c = 0
for _ in range(TEST_SAMPLE):
    i = random.randint(0, len(inputs) - 1)
    out = model(inputs[i]).detach().cpu().numpy()
    if get_idx(max_t(out), out) == get_idx(max_t(labels[i]), labels[i]):
        print(f"got {out} wanted {labels[i]} good")
        c += 1
    else:
        print(f"got {out} wanted {labels[i]} bad")

print(c / TEST_SAMPLE)


