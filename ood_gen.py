import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.autograd import grad
import torch.nn.functional as f

from sklearn.metrics import f1_score, accuracy_score

class Net(nn.Module):
    def __init__(self, larger=False):
        super(Net, self).__init__()

        self.layers = nn.ModuleList([])

        self.layers.append(nn.Linear(480, 606))
        if larger:
            print('more capacity model')
            # seems like too much capacity for IRM leads to plateau-ing lower
            # it would get stuck at ~30%, while less capacity model settles at ~70%
            # self.layers.append(nn.Linear(606, 909))
            # self.layers.append(nn.Linear(909, 1024))
            # self.layers.append(nn.Linear(1024, 606))
            self.layers.append(nn.Linear(606, 909))
            self.layers.append(nn.Linear(909, 606))
            
        else:
            print('less capacity model')
            self.layers.append(nn.Linear(606, 303))
            self.layers.append(nn.Linear(303, 606))
        self.layers.append(nn.Linear(606, 9))

        # for l in self.layers:
        #     l.weight.data.fill_(1.1)
        #     l.bias.data.fill_(1.1)

    def forward(self, x):
        # x = f.relu(self.fc1(x))
        # x = f.relu(self.fc2(x))
        # x = f.relu(self.fc3(x))
        # # x = f.softmax(self.fc4(x), dim=0)
        # x = self.fc4(x)

        for i, l in enumerate(self.layers):
            if not i == len(self.layers)-1:
                x = f.relu(l(x))
            else:
                x = l(x)
        return x


# env_p_xs = torch.tensor(env_p_xs)
# env_p_yxs = torch.tensor(env_p_yxs)
# x = torch.eye(3)

def bce_xy(model, x, y, w=1):

    ''' compute the loss for a given pair of envs for p_x and p_yx '''
    logits = model(x)
    # p_x = env_p_xs[env_x].view(-1, 1)
    # p_yx = env_p_yxs[env_y].view(-1, 1)
    # bce = torch.binary_cross_entropy_with_logits(w * logits, y, reduction='none')
    bce = torch.binary_cross_entropy_with_logits(w * logits, y)
    return bce

    # calc loss of model

    # # nll_loss = nn.NLLLoss()
    # loss_fn = nn.MSELoss()
    # return loss_fn(model(x.double()),y)


def rex_loss(model, env, penalty_weight):
    losses = [bce_xy(model, x_e, y_e).mean() for x_e, y_e in env]
    losses = torch.stack(losses)
    # return losses.sum() / penalty_weight + penalty_weight * losses.var(), [l.item() for l in losses]
    return (losses.sum() / (penalty_weight+1)) + penalty_weight * losses.var(), [l.item() for l in losses]

def erm_loss(model, env, penalty_weight):
    losses = [bce_xy(model, x_e, y_e).mean() for x_e, y_e in env]
    return torch.stack(losses).sum(), [l.item() for l in losses]

def irm_loss(model, env, penalty_weight):
    # w = torch.ones(1, requires_grad=True)
    # w = torch.nn.Parameter(torch.Tensor([1.0]))
    # losses = [bce_xy(model, x_e, y_e, w) for x_e, y_e in env]
    # losses = torch.stack(losses)

    # error = losses.mean().sum()

    # # penalty = grad(losses.sum(), w, create_graph=True)[0].pow(2).mean()

    # g1 = grad(losses[0::2].mean(), w, create_graph=True)[0]
    # g2 = grad(losses[1::2].mean(), w, create_graph=True)[0]
    # penalty = (g1 * g2).sum()

    # # return error + min(penalty_weight, 100) * penalty, [l.item() for l in losses]
    # return error + penalty_weight * penalty, [l.item() for l in losses]

    error = 0
    penalty = 0
    w = torch.nn.Parameter(torch.Tensor([1.0]))

    losses = []

    for x_e, y_e in env:
        loss_erm = bce_xy(model, x_e, y_e, w)

        losses.append(loss_erm.mean())

        g1 = grad(loss_erm[0::2].mean(), w, create_graph=True)[0]
        g2 = grad(loss_erm[1::2].mean(), w, create_graph=True)[0]
        penalty += (g1 * g2).sum()

        error += loss_erm.mean()

    return error + penalty_weight * penalty, [l.item() for l in losses]


loss_fcts = {
    "REx": rex_loss,
    "IRM": irm_loss,
    "ERM": erm_loss,
    }

# envs = env_split(X, Y, P)
# x_e, y_e = envs[e]
def env_split(X, Y, P):
    env = {}
    
    for i, p in enumerate(P):
        if not p in env:
            env[p]=[[],[]]
        env[p][0].append(X[i])
        env[p][1].append(Y[i])

    return [(torch.tensor(x_e, requires_grad=True), torch.tensor(y_e)) for x_e, y_e in list(env.values())]

def eval(model, env):
    loss = []
    f1 = []
    acc = []

    for x_e, y_e in env:
        l = bce_xy(model, x_e, y_e).mean()

        y_h = model(x_e)

        f = f1_score(np.argmax(y_h.detach(), axis=1), np.argmax(y_e.detach(), axis=1), average='weighted')
        a = accuracy_score(np.argmax(y_h.detach(), axis=1), np.argmax(y_e.detach(), axis=1))

        loss.append(l)
        f1.append(f)
        acc.append(a)

    return torch.tensor(loss).mean(), torch.tensor(f1).mean(), torch.tensor(acc).mean()

def inter_subj_ood(e_tr, e_te, epochs=50):

    run_methods = ["REx", "IRM", "ERM"]

    models = []

    xs = []
    ys = []
    zs = []

    for method in run_methods:
        print('Running model: '+method)

        if method in ['IRM', 'REx']:
            model = Net(True).double()
        else:
            model = Net().double()

        optimizer = optim.Adam(model.parameters(), lr=0.001) # match optimizer?
        loss_fct = loss_fcts[method]

        for step in range(epochs):

            if method == 'IRM':
                # penalty_weight = step+1
                # penalty_weight = (step+1)**1.2
                penalty_weight = (step+1)**1.6
            elif method == 'REx':
                penalty_weight = step**2
            else:
                penalty_weight = 0

            loss, z = loss_fct(model, e_tr, penalty_weight=penalty_weight)

            zs.extend(z)
            xs.extend(list(range(len(z))))
            ys.extend([step]*len(z))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, f1, acc = eval(model, e_tr)
            print('Epoch: '+ str(step)+', Loss: '+str(loss.item())+', F1: '+str(f1.item())+', Acc: '+str(acc.item()))

        model.eval()
        test_loss, test_f1, test_acc = eval(model, e_te)

        print('Test loss: '+ str(test_loss.item()))
        print('Test F1: '+ str(test_f1.item()))
        print('Test Acc: '+ str(test_acc.item()))

        models.append(model)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ys,xs,zs)
        plt.show()

        print('+'*30)

    return models
