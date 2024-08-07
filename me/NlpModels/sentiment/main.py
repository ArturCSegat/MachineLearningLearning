import time
import pandas as pd
import random
from gensim.models.word2vec import Word2Vec as w2v
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import get_tokenizer
from torch.utils.data import TensorDataset, DataLoader



def get_feeling(out: torch.Tensor) -> str:
    b = out.argmax()
    return str(b)

CONTEXT_SIZE = 10
EMBEDDING_DIM = 128
EPOCHS = 5
CBOW_LR = 0.05
LEARN_R = 0.025

def get_train_set(ammount: int=99999):
    df = pd.read_csv("data.csv", sep=",", encoding = "cp1252")
    tokenizer = get_tokenizer("basic_english")
    largest_tweet = 0
    
    data = df["text"]
    label = df["sentiment"]
    
    data_ts = []
    label_ts = []
    
    i = 0
    for d, l in zip(data, label):
        # print(f"d: {d} l: {l}")
        if i == ammount:
            break
    
        try:
            tokens = tokenizer(d)
            if len(tokens) > largest_tweet:
                largest_tweet = len(tokens)
            data_ts.append(tokens)
            o = [0.0, 0.0, 1.0]
            if l == "negative":
                o = [1.0, 0.0, 0.0]
            elif l == "neutral":
                o = [0.0, 1.0, 0.0]
            label_ts.append(o)
        except:
            continue
        i += 1

    return data_ts, torch.tensor(label_ts), largest_tweet

def embed_sentence(model: w2v, sentence: list[str], token_count: int, emb_dim: int) -> torch.Tensor:
    embedding = torch.cat([torch.tensor(model.wv.get_vector(w)) for w in sentence])
    if len(embedding) / emb_dim < token_count:
        missing = token_count - (len(embedding) // emb_dim)
        padding = torch.zeros(missing  * emb_dim)
        embedding = torch.cat((embedding, padding))
    return embedding

def embed_data_set(model: w2v, sentences: list[list[str]], sentence_len: int, emb_dim:  int) -> torch.Tensor:
    return torch.stack([embed_sentence(model, sentence, sentence_len, emb_dim) for sentence in sentences])


class SentimentModel(nn.Module):
    def __init__(self, embeddings: w2v, width, emb_dim=EMBEDDING_DIM):
        super(SentimentModel, self).__init__()
        self.embeddings = embeddings
        self.emb_dim = emb_dim
        self.width = width
        self.layers = nn.Sequential(
            nn.Linear(width * emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(),
        )
    def forward(self, inp: torch.Tensor):
        return self.layers(inp)

def train(model: SentimentModel, dataloader: DataLoader, epochs, loss_fn: nn.CrossEntropyLoss, optmizer: optim.SGD): 
    for i in range(epochs):
        i_loss = 0
        for data, label in dataloader:
            model.zero_grad()
            
            # print(f"word: {data[0]}")
            # print(f"embed: {model.embeddings.wv.get_vector(data[0])}")

            # print(f"data shape: {data.shape}")
            logits = model(data)
            # print(f"logits: {logits}, label: {label}")
            # print(f"logits: {logits.shape}, label: {label.shape}")
            # exit()

            loss = loss_fn(logits, label)
            i_loss += loss.item()
            loss.backward()
            optmizer.step()
        print(f"loss: {i_loss / len(dataloader)}")

data, labels, width = get_train_set()
sm = SentimentModel(w2v(sentences=data, window=CONTEXT_SIZE, alpha=CBOW_LR, shrink_windows=False, min_alpha=CBOW_LR, vector_size=EMBEDDING_DIM, min_count=0), width)

print(f"width: {width}")

dataset = TensorDataset(embed_data_set(sm.embeddings, data, width, EMBEDDING_DIM), labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
t1 = time.time()
train(sm, dataloader, EPOCHS, nn.CrossEntropyLoss(), optim.SGD(sm.parameters(), LEARN_R))
print()
print(f"done training in {time.time() - t1}")

def get_sentiment(out: torch.Tensor) -> str:
    b = out.argmax().item()
    if b == 0:
        return f"{out}: negative"
    elif b == 1:
        return f"{out}: neutral"
    return f"{out}: positive"

print()
test = ["i", "love", "you"]
print(f"{test}\nprediction: {get_sentiment(sm(embed_sentence(sm.embeddings, test, sm.width, sm.emb_dim)))}:\t real: [0, 0, 1]")
test = ["i", "hate", "you"]
print(f"{test}\nprediction: {get_sentiment(sm(embed_sentence(sm.embeddings, test, sm.width, sm.emb_dim)))}\t real: [1, 0, 0]")
test = ["you", "read", "a", "book"]
print(f"{test}\nprediction: {get_sentiment(sm(embed_sentence(sm.embeddings, test, sm.width, sm.emb_dim)))}\t real: [0, 1, 0]")
print()
print()
print()
while True:
    entry = input(f"enter a frase, max token lenght: {width}\t")
    entry = get_tokenizer("basic_english")(entry)
    try:
        print(f"{entry}\nprediction: {get_sentiment(sm(embed_sentence(sm.embeddings, entry, sm.width, sm.emb_dim)))}")
    except:
        print("processing error")
    print()

