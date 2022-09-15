import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# from torchvision.transforms import ToTensor

from load_data import load_surface_data, load_raw_surface_data

np.random.seed(0)
torch.manual_seed(0)

class ViT(nn.Module):
	def __init__(self, input_shape=(100,5,6), patch_size=(5,1,2), hidden_d=8, n_heads=2, out_d=9):
		super(ViT, self).__init__()

		self.input_shape = input_shape
		self.patch_size = patch_size

		self.n_patches = int((input_shape[0]/patch_size[0])*(input_shape[1]/patch_size[1])*(input_shape[2]/patch_size[2]))
		self.input_d = int(patch_size[0]*patch_size[1]*patch_size[2])

		self.hidden_d = hidden_d

		self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

		self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

		self.ln1 = nn.LayerNorm((self.n_patches+ 1, self.hidden_d))

		self.msa = MSA(self.hidden_d, n_heads)

		self.ln2 = nn.LayerNorm((self.n_patches+ 1, self.hidden_d))

		self.enc_mlp = nn.Sequential(
			nn.Linear(self.hidden_d, self.hidden_d),
			nn.ReLU()
		)

		self.mlp = nn.Sequential(
			nn.Linear(self.hidden_d, out_d),
			nn.Softmax(dim=-1)
		)

	def forward(self, images):
		# n, w, h, c = images.shape
		n = images.shape[0]

		patches = images.reshape(n, self.n_patches, self.input_d)

		tokens = self.linear_mapper(patches)

		tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

		tokens += get_positional_embeddings(self.n_patches+ 1, self.hidden_d).repeat(n,1,1)

		out = tokens + self.msa(self.ln1(tokens))

		out = out + self.enc_mlp(self.ln2(out))

		out = out[:,0]

		return self.mlp(out)


class MSA(nn.Module):
	def __init__(self, d, n_heads=2):
		super(MSA, self).__init__()
		self.d = d
		self.n_heads = n_heads

		d_head = int(d/n_heads)

		self.q_mappings = [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
		self.k_mappings = [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
		self.v_mappings = [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
		self.d_head = d_head
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, sequences):
		result = []

		for sequence in sequences:
			seq_result = []
			for head in range(self.n_heads):
				q_mappings = self.q_mappings[head]
				k_mappings = self.k_mappings[head]
				v_mappings = self.v_mappings[head]

				seq = sequence[:, head * self.d_head: (head+1)*self.d_head]
				q,k,v = q_mappings(seq), k_mappings(seq), v_mappings(seq)

				attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
				seq_result.append(attention @ v)
			result.append(torch.hstack(seq_result))
		return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


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


def main(seed=39):
	epochs = 250
	lr = 0.01
	batch_size = 16

	# X_tr, Y_tr, X_te, Y_te, P_tr, P_te = load_raw_surface_data(seed, True)
	X_tr, Y_tr, P_tr, X_te, Y_te, P_te, _ = load_surface_data(seed, True)

	# e_tr = env_split(torch.tensor(X_tr), Y_tr, P_tr)
	# e_te = env_split(X_te, Y_te, P_te)

	X_tr = torch.tensor(X_tr.astype(np.float32))
	Y_tr = torch.tensor(Y_tr)
	X_te = torch.tensor(X_te.astype(np.float32))
	Y_te = torch.tensor(Y_te)

	train_dataset = TensorDataset(X_tr,Y_tr)
	test_dataset = TensorDataset(X_te,Y_te)

	train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
	test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

	# model = ViT()
	model = ViT(input_shape=(480,1,1), patch_size=(5,1,1), n_heads=4)

	# x = torch.rand(3,100,5,6)
	# y = torch.rand(3,9)

	# y_hat = model(x)

	# print(torch.argmax(y_hat, dim=))

	print("Loaded data and model")

	optimizer = Adam(model.parameters(), lr=lr)
	criterion = CrossEntropyLoss()

	for epoch in range(epochs):
		model.train()
		train_loss = 0.0

		for X,Y in train_loader:
			y_hat = model(X)
			loss = criterion(y_hat, Y)/len(X)

			train_loss += loss.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(f"Epoch {epoch+1}/{epochs} loss: {train_loss:.2f}")


		model.eval()
		test_loss = 0.0

		for X,Y in test_loader:
			y_hat = model(X)
			loss = criterion(y_hat, Y)/len(X)

			test_loss += loss.item()

		print(f"Test loss: {test_loss:.2f}")


	correct, total = 0,0

	test_loss = 0.0

	for X,Y in test_loader:
		y_hat = model(X)
		loss = criterion(y_hat, Y)/len(X)

		test_loss += loss.item()

		correct += torch.sum(torch.argmax(y_hat, dim=1)==torch.argmax(Y, dim=1)).item()
		total += len(X)

	print('Final Evaluation:')
	print(f"Test loss: {test_loss:.2f}")
	print(f"Test accuracy: {correct / total * 100:.2f}%")


def get_positional_embeddings(sequence_length, d):
	result = torch.ones(sequence_length, d)
	for i in range(sequence_length):
		for j in range(d):
			result[i][j] = np.sin(i/(10000 ** (j/d))) if j%2==0 else np.cos(i/(10000 ** (j-1)/d))
	return result


if __name__ == '__main__':
	# model = ViT()
	# # model = ViT((480,1,1),(5,1,1))

	# x = torch.rand(3,100,5,6)
	# # x = torch.rand(3,480,1,1)

	# print(model(x).shape)

	main()