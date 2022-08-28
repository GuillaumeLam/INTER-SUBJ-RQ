import numpy as np
import torch
import pickle
import os
from random import seed, randint
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.autograd import grad
import torch.nn.functional as f

from sklearn.decomposition import PCA

from sklearn.metrics import f1_score, accuracy_score

from load_data import load_surface_data

class Net(nn.Module):
    def __init__(self, size):
        super(Net, self).__init__()

        self.layers = nn.ModuleList([])

        if size == 'large':
            # seems like too much capacity for IRM leads to plateau-ing lower
            # it would get stuck at ~30%, while less capacity model settles at ~70%
            # self.layers.append(nn.Linear(606, 909))
            # self.layers.append(nn.Linear(909, 1024))
            # self.layers.append(nn.Linear(1024, 606))
            self.layers.append(nn.Linear(480, 909))
            self.layers.append(nn.Linear(909, 606))
            self.layers.append(nn.Linear(606, 909))
            self.layers.append(nn.Linear(909, 9))
            
        elif size == 'standard':
            self.layers.append(nn.Linear(480, 606))
            self.layers.append(nn.Linear(606, 303))
            self.layers.append(nn.Linear(303, 606))
            self.layers.append(nn.Linear(606, 9))

        elif size == 'small':
            self.layers.append(nn.Linear(480, 404))
            self.layers.append(nn.Linear(404, 202))
            self.layers.append(nn.Linear(202, 404))
            self.layers.append(nn.Linear(404, 9))

    def forward(self, x):
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
    # loss = []
    f1 = []
    acc = []

    for x_e, y_e in env:
        # l = bce_xy(model, x_e, y_e).mean()

        y_h = model(x_e)

        f = f1_score(np.argmax(y_h.detach(), axis=1), np.argmax(y_e.detach(), axis=1), average='weighted')
        a = accuracy_score(np.argmax(y_h.detach(), axis=1), np.argmax(y_e.detach(), axis=1))

        # loss.append(l)
        f1.append(f)
        acc.append(a)

    return torch.tensor(f1).mean(), torch.tensor(acc).mean()

def inter_subj_ood(run_method, e_tr, e_te, seed=39, model_size='standard', epochs=50, lr=0.001, penalty_weight_factor=1.6):
    torch.manual_seed(seed)

    print('SETTINGS Epoch:'+str(epochs)+', Learning Rate:'+str(lr)+', Penalty Weight Factor:'+str(penalty_weight_factor)+', Model Size:'+model_size)

    # run_methods = ["REx", "IRM", "ERM"]
    # run_methods = [run_method]
    metrics = ['risk', 'loss', 'f1', 'acc']

    models = []

    xs = []
    ys = []
    zs = []

    inter_subj_gap = {}

    for m in metrics:
        inter_subj_gap[m] = ([],[],[]) # [0] for train metric, [1] for test metric, [2] for gap (train-test)
        # inter_subj_gap['IRM'][m] = ([],[],[])
        # inter_subj_gap['ERM'][m] = ([],[],[])

    print('Running model: '+run_method)

    model = Net(model_size).double()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fct = loss_fcts[run_method]

    for step in range(epochs):

        # TRAIN
        penalty_weight = step**penalty_weight_factor

        model.train()

        risk, z = loss_fct(model, e_tr, penalty_weight=penalty_weight)

        zs.extend(z)
        xs.extend(list(range(len(z))))
        ys.extend([step]*len(z))

        optimizer.zero_grad()
        risk.backward()
        optimizer.step()

        f1, acc = eval(model, e_tr)
        print('Epoch: '+ str(step))
        print('Risk: '+str(risk.item())+',      Loss: '+str(np.array(z).mean())+',      F1: '+str(f1.item())+',      Acc: '+str(acc.item()))

        # TEST
        model.eval()

        test_risk, test_z = loss_fct(model, e_te, penalty_weight=penalty_weight)

        test_f1, test_acc = eval(model, e_te)

        print('Test Risk: '+str(test_risk.item())+', Test Loss: '+str(np.array(test_z).mean())+', Test F1: '+str(test_f1.item())+', Test Acc: '+str(test_acc.item()))

        inter_subj_gap['risk'][0].append(risk.item())
        inter_subj_gap['risk'][1].append(test_risk.item())
        inter_subj_gap['risk'][2].append(risk.item()-test_risk.item())

        inter_subj_gap['loss'][0].append(np.array(z).mean())
        inter_subj_gap['loss'][1].append(np.array(test_z).mean())
        inter_subj_gap['loss'][2].append(np.array(z).mean()-np.array(test_z).mean())

        inter_subj_gap['f1'][0].append(f1.item())
        inter_subj_gap['f1'][1].append(test_f1.item())
        inter_subj_gap['f1'][2].append(f1.item()-test_f1.item())

        inter_subj_gap['acc'][0].append(acc.item())
        inter_subj_gap['acc'][1].append(test_acc.item())
        inter_subj_gap['acc'][2].append(acc.item()-test_acc.item())

    # models.append(model)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ys,xs,zs)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Environment ie. patient')
    ax.set_zlabel('Risk')
    plt.show()

    print('+'*30)

    return model, inter_subj_gap

def wrap(seed, run_method, config):
    print('+='*25)
    print('Seed:', seed)
    config_name = 'model_size=' +config['ms']+ ',epochs=' +str(config['epochs'])+ ',lr=' +str(config['lr'])+ ',penalty_weight_factor=' +str(config['pwf'])
    save_path = 'out/'+run_method+'/'+config_name+'/seed='+str(seed)+'/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    X_tr, Y_tr, P_tr, X_te, Y_te, P_te, _ = load_surface_data(seed, True)

    e_tr = env_split(X_tr, Y_tr, P_tr)
    e_te = env_split(X_te, Y_te, P_te)

    # def ood_train(config):
    #     models, isg = inter_subj_ood(run_method, e_tr, e_te, seed=seed, model_size=config['ms'], epochs=config['epochs'], lr=config['lr'], penalty_weight_factor=config['pwf'])
    #     tune.report(f1_gap=) # report highest f1 score with lowest f1 inter-subj gap with tol=t
    #     # return isg
    # return ood_train

    _, isg = inter_subj_ood(run_method, e_tr, e_te, seed=seed, model_size=config['ms'], epochs=config['epochs'], lr=config['lr'], penalty_weight_factor=config['pwf'])

    # SAVE resulting run data
    pickle.dump(isg, open(save_path+'ood_run_data('+config_name+').pkl','wb'))

    graph_inter_subj_gap(isg, run_method, save_path=save_path, config_name=config_name)
    print('+='*25)

def ood_fold(cv):
    seed(39)
    seeds = [randint(0,1000) for _ in range(0,cv)]

    run_methods = ["REx", "IRM", "ERM"]

    lr = [0.0001, 0.001, 0.01]
    epochs = [50, 100]
    pwf = [0.8, 1.2, 1.6]
    ms = ['small', 'standard', 'large']

    # lr = [0.01]
    # epochs = [50]
    # pwf = [1.6]
    # ms = ['standard']

    config = {}
    
    for method in run_methods:
        for l in lr:
            for e in epochs:
                for p in pwf:
                    for m in ms:
                        config['lr'] = l
                        config['epochs'] = e
                        config['pwf'] = p
                        config['ms'] = m

                        for s in seeds:
                            wrap(s, method, config)
            

def graph_inter_subj_gap(inter_subj_gap, run_method, save_path=None, config_name=None):
    plt.clf()

    scatter = False

    if scatter:
        m = plt.scatter(inter_subj_gap['f1'][0], inter_subj_gap['f1'][2], marker='o')
        # irm = plt.scatter(inter_subj_gap['IRM']['f1'][0], inter_subj_gap['IRM']['f1'][2], marker='x')
        # erm = plt.scatter(inter_subj_gap['ERM']['f1'][0], inter_subj_gap['ERM']['f1'][2], marker='p')
        plt.legend((m), (run_method), loc='upper left')
    else:
        rex = plt.plot(inter_subj_gap['f1'][0], inter_subj_gap['f1'][2], marker='o', label=run_method)
        # irm = plt.plot(inter_subj_gap['IRM']['f1'][0], inter_subj_gap['IRM']['f1'][2], marker='x', label='IRM')
        # erm = plt.plot(inter_subj_gap['ERM']['f1'][0], inter_subj_gap['ERM']['f1'][2], marker='p', label='ERM')
        plt.legend()
    plt.xlabel('Training F1')
    plt.ylabel('Inter-subject gap F1')
    plt.title(run_method+' Inter-subject gap vs Training F1')

    if save_path is None and config_name is None:
        plt.show()
    else:
        plt.savefig(save_path+'inter_subj_gap_vs_train('+config_name+').png')


    # for method in ['REx', 'IRM', 'ERM']:
    plt.clf()
    x = range(len(inter_subj_gap['loss'][0]))
    plt.plot(x, inter_subj_gap['loss'][0], label='training')
    plt.plot(x, inter_subj_gap['loss'][1], label='test')
    plt.legend()
    plt.title(run_method+' loss')
    if save_path is None and config_name is None:
        plt.show()
    else:
        plt.savefig(save_path+'train-test_loss('+config_name+').png')

    plt.clf()
    x = range(len(inter_subj_gap['risk'][0]))
    plt.plot(x, inter_subj_gap['risk'][0], label='training')
    plt.plot(x, inter_subj_gap['risk'][1], label='test')
    plt.legend()
    plt.title(run_method+' risk')
    if save_path is None and config_name is None:
        plt.show()
    else:
        plt.savefig(save_path+'train-test_risk('+config_name+').png')

# %matplotlib inline
def graph_env_pca(env, patient_id):
    X,Y = env[patient_id]

    plt.rcParams['figure.figsize'] = [10, 10]

    pca = PCA().fit(X.detach().numpy())
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    print('First 5 component variation accountability:',np.cumsum(pca.explained_variance_ratio_)[:5])
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.grid()
    plt.show()

    pca = PCA(2)  # project from 480 to 2 dimensions
    projected = pca.fit_transform(X.detach().numpy())
    plt.scatter(projected[:, 0], projected[:, 1],
                c=Y.detach().numpy().argmax(axis=1), edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Spectral', 9))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar();
    plt.show()

# %matplotlib notebook
def graph_env_pca_3d(env, patient_id):
    X,Y = env[patient_id]

    plt.rcParams['figure.figsize'] = [8, 8]

    pca = PCA(3)  # project from 480 to 3 dimensions
    projected = pca.fit_transform(X.detach().numpy())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                c=Y.detach().numpy().argmax(axis=1), edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Spectral', 9))
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.set_zlabel('component 3')
    plt.show()

# def graph_label_pca_3d(env):
#     label = {}

#     for patient_id,(X,Y) in enumerate(env):
#         # label[patient_id] = [] 
#         pca = PCA(3)  # project from 480 to 3 dimensions
#         projected = pca.fit_transform(X.detach().numpy())

#         for i in range(len(Y)):
#             key = Y[i].argmax()
#             if key in label:
#                 label[key].append(projected[i,:])