import torch
import torch.nn as nn
import torch.optim as optim
import random
from math_helper import arr_from


def training_data(num_samples):
  data = []
  labels = []
  for _ in range(num_samples):
    input_data = [random.uniform(0, 100) for _ in range(3)]
    input_data[1] = 1

    label = sum(input_data) - 1
    data.append(input_data)
    labels.append([label])
  return torch.tensor(data), torch.tensor(labels)

class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
                    nn.Linear(3, 1),
                    nn.ReLU(),
                    nn.Linear(1, 1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

model = CustomNetwork()

loss_fn = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.00001)

# Prepare training data
inputs, labels = training_data(500_000)

# Train the model
epochs = 20
model.train()
for epoch in range(epochs):
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Test the trained model
test_inputs = [arr_from(1, 1, 2), arr_from(2, 1, 5)]
for test_input in test_inputs:
    test_input_tensor = torch.tensor(test_input, dtype=torch.float32)
    predicted_output = model(test_input_tensor).detach().cpu().numpy()
    print(f"Input: {test_input}, Predicted Output: {predicted_output}")

