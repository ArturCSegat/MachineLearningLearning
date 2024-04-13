import torch
import random
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchtext.data import get_tokenizer
import gensim.downloader as gd
import time
from gensim.models.word2vec import Word2Vec as w2v

EMBEDDING_DIM = 100
CONTEXT_SIZE = 10
EPOCHS = 5
LEARN_R = 0.001

def ngrams(sentences):
    ngrams = []
    for s in sentences:
        if len(s) <= CONTEXT_SIZE:
            continue
        for i in range(CONTEXT_SIZE // 2, len(s) - CONTEXT_SIZE // 2):
            ngrams.append(
                (
                    [s[i - (CONTEXT_SIZE // 2) + j] for j in range(CONTEXT_SIZE // 2)] + [s[i + j + 1] for j in range(CONTEXT_SIZE // 2)],
                    s[i]
                )
            )
    return ngrams

def get_train_set(ammount: int=99999):
    df = pd.read_csv("data.csv", sep=",", encoding = "cp1252")
    tokenizer = get_tokenizer("basic_english")
    largest_tweet = 0
    vocab = set()
    
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
            t = tokenizer(d)
            for tok in t:
                vocab.add(tok)
            data_ts.append(t)
            o = [0.0, 0.0, 1.0]
            if l == "negative":
                o = [1.0, 0.0, 0.0]
            elif l == "neutral":
                o = [0.0, 1.0, 0.0]
            label_ts.append(o)
        except:
            continue
        i += 1

    
    token_to_idx = {w: i  for w, w, in enumerate(vocab)}
    return data_ts, torch.tensor(label_ts), vocab, token_to_idx, largest_tweet


# wv = gd.load('text8')
# t1 = time.time()
# print(f"done loading in {time.time() - t1}")

sentences, _, _, _, _ = get_train_set(500)

t1 = time.time()
gmodel = w2v(sentences, vector_size=EMBEDDING_DIM, epochs=EPOCHS, alpha=LEARN_R, window=CONTEXT_SIZE, shrink_windows=False, min_alpha=LEARN_R)
print(f"done training in {time.time() - t1}")
print(f"{gmodel.wv.get_vector('the')}")
print()
print()

class Cbow(nn.Module):
    def __init__ (self, vocab_size, embed_dim, context_size, tok_to_idx, loss):
        super(Cbow, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.tok_to_idx = tok_to_idx
        self.loss = loss
        self.layers = nn.Sequential(
            nn.Linear(context_size * embed_dim, 300), 
            nn.ReLU(),
            nn.Linear(300, vocab_size),
            nn.LogSoftmax(),
        )
    def forward(self, inp):
        inp = self.embeddings(inp).view((1, -1))
        return self.layers(inp)

def run_epochs(model, ngrams: list[tuple[list[str], str]], epochs: int, opt):
    for _ in range(epochs):
        random.shuffle(ngrams)
        avg_l = 0
        for context, label in ngrams:
            idxs = torch.tensor([model.tok_to_idx[t] for t in context], dtype=torch.long)
            model.zero_grad()
            logits = model(idxs)
            l = model.loss(logits, torch.tensor([model.tok_to_idx[label]], dtype=torch.long))
            avg_l += l.item()
            l.backward()
            opt.step()
        # print(f"loss: {avg_l/len(ngrams)}")


train_toks, label_toks, vocab, tok_idx, max_len = get_train_set(500)
cbow = Cbow(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, tok_idx, nn.NLLLoss())
t1 = time.time()
n = ngrams(train_toks)
print(f"done ngramming in {time.time() - t1}")
t1 = time.time()
run_epochs(cbow, n, EPOCHS, optim.SGD(cbow.parameters(), LEARN_R))
print(f"done training second model in {time.time() - t1}")
print()
print()

# test = n[69]
# print(test[0][:CONTEXT_SIZE // 2])
# logits = cbow(torch.tensor([tok_idx[w] for w in test[0]]))
# print(f"predict: {list(tok_idx.keys())[torch.argmax(logits)]}")
# print(test[0][CONTEXT_SIZE // 2:])
# print(f"real: {test[1]}")
# print()

def w(s: str) -> torch.Tensor:
    return cbow.embeddings.weight[tok_idx[s]]

print(w("the"))
        
