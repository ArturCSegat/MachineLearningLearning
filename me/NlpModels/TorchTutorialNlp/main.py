import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import get_tokenizer
import random

tknizer = get_tokenizer("basic_english")

CONTEXT_SIZE = 3
EMBEDDING_DIM = 50
DATA_SET_SIZE = 1000

# Tokenize each line separately
tokenized_sentences = []
text = "" 
with open("/home/arturcs/Downloads/chat.txt", "r") as f:
    i = 0
    line = f.readline()
    while line != "" and i < DATA_SET_SIZE:
        i += 1
        text += line.strip()
        line = f.readline()
# Combine all tokenized sentences into one list
test_sentence = tknizer(text)
print("tokenized done")

ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
# ngrams = [
#     (
#         [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE // 2)] + [test_sentence[i + j + 1] for j in range(CONTEXT_SIZE // 2)],
#         test_sentence[i]
#     )
#     for i in range(CONTEXT_SIZE // 2, len(test_sentence) - CONTEXT_SIZE // 2)
# ]

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 2048)
        self.linear2 = nn.Linear(2048, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(20):
    total_loss = 0
    random.shuffle(ngrams)  # Shuffle data before each epoch
    for context, target in ngrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)

print(losses)  # The loss decreased every iteration over the training data!
print()
print()
print()

inps = ["artur-cs", "oii", "amor"]
idxs = torch.tensor([word_to_ix[w] for w in inps])
model.zero_grad()
outs = model(idxs)
idx = torch.argmax(outs)

p = list(word_to_ix.keys())[idx]
print(p)

