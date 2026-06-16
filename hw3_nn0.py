#111210552 林小蓮

import nn0
import random
import math

def sigmoid(x):
    neg_x = x * Value(-1.0)
    return Value(1.0) / (Value(1.0) + neg_x.exp())

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return sigmoid(act)

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self):
        self.layer1 = Layer(2, 4)
        self.layer2 = Layer(4, 1)

    def __call__(self, x):
        hidden = self.layer1(x)
        output = self.layer2(hidden)
        return output[0]

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()

data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

def train():
    model = MLP()
    params = model.parameters()
    optimizer = nn0.Adam(params, lr=0.1)
    num_steps = 500

    for step in range(num_steps):
        total_loss = Value(0.0)
        for x_vals, y_true in data:
            x_input = [Value(v) for v in x_vals]
            y_pred = model(x_input)
            y_target = Value(y_true)
            eps = Value(1e-7)
            loss_i = (
                Value(-1.0) * (
                    y_target * (y_pred + eps).log() +
                    (Value(1.0) - y_target) * (Value(1.0) - y_pred + eps).log()
                )
            )
            total_loss = total_loss + loss_i

        avg_loss = total_loss * Value(1.0 / len(data))
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == num_steps - 1:
            print(f"Step {step:4d} | Loss: {avg_loss.data:.6f}")

    return model

def evaluate(model):
    correct = 0
    for x_vals, y_true in data:
        x_input = [Value(v) for v in x_vals]
        y_pred = model(x_input)
        pred_label = 1 if y_pred.data >= 0.5 else 0
        expected_label = int(y_true)
        is_correct = pred_label == expected_label
        if is_correct:
            correct += 1
        status = "✓" if is_correct else "✗"
        print(f"x={x_vals}  →  {y_pred.data:.4f} ({pred_label})    {expected_label}      {status}")
    print(f"Accuracy: {correct}/{len(data)} = {correct/len(data)*100:.1f}%")

if __name__ == "__main__":
    random.seed(42)
    model = train()
    evaluate(model)
