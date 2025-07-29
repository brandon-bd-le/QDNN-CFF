import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

fns = F1F2()
calc = F_calc()

torch.manual_seed(42)

class ClassicalDNN(nn.Module):
    def __init__(self):
        super(ClassicalDNN, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(3, 64))
        for _ in range(8):
            self.fc_layers.append(nn.Linear(64, 64))
        self.fc_layers.append(nn.Linear(64, 4))

    def forward(self, inputs):
        output = inputs
        for layer in self.fc_layers[:-1]:
            output = torch.relu(layer(output))
        outputs = self.fc_layers[-1](output)
        return outputs

def modelf(phi_x, QQ, x_b, t, k, ReH, ReE, ReHt, dvcs):
    F_1, F_2 = fns.f1_f21(t)
    kins = [phi_x, QQ, x_b, t, k, F_1, F_2]
    cffs = [ReH, ReE, ReHt, dvcs]
    return calc.fn_1(kins, cffs)

def custom_loss(predicted_cffs, true_F, x_b, QQ, t, k, phi_x):
    ReH, ReE, ReHt, dvcs = torch.split(predicted_cffs, 1, dim=1)
    predicted_F = modelf(phi_x, QQ, x_b, t, k, ReH, ReE, ReHt, dvcs)
    loss = torch.mean((predicted_F - true_F.view_as(predicted_F)) ** 2)
    return loss

results = {
    'Set #': [], 'Replica #': [],
    'Pred ReH': [],
    'Pred ReE': [],
    'Pred ReHt': [],
    'Pred dvcs': [],
}

for j in range(len(error_bins)):
    model = ClassicalDNN()
    optimizer = optim.RMSprop(model.parameters(), lr=0.005)
    
    phi_x = torch.tensor(error_bins[j]['phi_x'].values, dtype=torch.float32)
    QQ = torch.tensor(error_bins[j].iloc[0]['QQ'], dtype=torch.float32)
    x_b = torch.tensor(error_bins[j].iloc[0]['x_b'], dtype=torch.float32)
    t = torch.tensor(error_bins[j].iloc[0]['t'], dtype=torch.float32)
    k = torch.tensor(error_bins[j].iloc[0]['k'], dtype=torch.float32)
    true_F = torch.tensor(error_bins[j]['F'].values, dtype=torch.float32)

    X = torch.stack([x_b.unsqueeze(0), QQ.unsqueeze(0), t.unsqueeze(0)], dim=1).float()

    epochs = 500
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predicted_cffs = model(X)
        loss = custom_loss(predicted_cffs, true_F, x_b, QQ, t, k, phi_x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print("Training complete for bin:", i, j)
    print(f'Epoch {epoch}, Loss: {loss.item()}')
    print(predicted_cffs)
    
    pred_params = [predicted_cffs[:, 0].item(), predicted_cffs[:, 1].item(),
        predicted_cffs[:, 2].item(), predicted_cffs[:, 3].item()]

    results['Set #'].append(i+1)
    results['Replica #'].append(j+1)
    results['Pred ReH'].append(pred_params[0])
    results['Pred ReE'].append(pred_params[1])
    results['Pred ReHt'].append(pred_params[2])
    results['Pred dvcs'].append(pred_params[3])
