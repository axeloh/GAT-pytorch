import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import time
import os
import numpy as np


from utils import print_info_about_dataset, plot_dataset, accuracy, create_adjacency_matrix
from models import GAT


def train(model, optimizer, x, y, A, train_mask, val_mask, n_epochs, plot=False, device=None):
	train_accuracies, val_accuracies = [], []
	train_losses, val_losses = [], []
	start = time.time()

	for epoch in range(n_epochs):
		model.train()
		optimizer.zero_grad()
		out = model(x, A)
		train_loss = F.cross_entropy(out[train_mask], y[train_mask])
		train_acc = accuracy(out[train_mask], y[train_mask])
		train_loss.backward()
		optimizer.step()

		# Evaluate on validation set
		with torch.no_grad():
			model.eval()
			out = model(x, A)
			val_loss = F.cross_entropy(out[val_mask], y[val_mask])
			val_acc = accuracy(out[val_mask], y[val_mask])

		train_accuracies.append(train_acc.item())
		train_losses.append(train_loss.item())
		val_accuracies.append(val_acc.item())
		val_losses.append(val_loss.item())

		print(f'Epoch: {epoch+1}, '
			  f'train_loss: {train_loss.item():.3f}, '
			  f'train_acc: {train_acc.item():.3f}, '
			  f'val_loss: {val_loss.item():.3f} '
			  f'val_acc: {val_acc.item():.3f} ',
			  )

	print(f'Training done in {(time.time() - start):.1f}s')

	if plot:
		if not os.path.exists('outputs'):
			os.makedirs('outputs')

		plt.plot(train_losses, label="Train losses")
		plt.plot(val_losses, label="Validation losses")
		plt.xlabel("# Epoch")
		plt.ylabel("Loss")
		plt.legend(loc='upper right')
		plt.savefig('./outputs/att_loss_plot.png')
		plt.show()
		plt.close()

		plt.plot(train_accuracies, label="Train accuracy")
		plt.plot(val_accuracies, label="Validation accuracy")
		plt.xlabel("# Epoch")
		plt.ylabel("Accuracy")
		plt.legend(loc='upper right')
		plt.savefig('./outputs/att_accuracy_plot.png')
		plt.show()
		plt.close()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Device: {device}')

	dataset = Planetoid(root='/tmp/Cora', name='Cora')  # Cora, CiteSeer, or PubMed
	print(dataset.data)
	print_info_about_dataset(dataset)

	data = dataset.data
	num_nodes = data.num_nodes
	num_features = dataset.num_node_features

	x = data.x.to(device)  # Node features
	y = data.y.to(device)  # Node classes

	num_targets = len(torch.unique(y))
	print(f'Num classes: {num_targets}')

	train_mask = data.train_mask.to(device)
	val_mask = data.val_mask.to(device)

	A = create_adjacency_matrix(num_nodes, data.edge_index, device=device)

	model = GAT(
		node_dim=num_features,
		hid_dim=8,
		num_classes=num_targets,
		dropout=0.6,
		alpha=0.2,
		num_heads=8
	)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

	if torch.cuda.is_available():
		model.cuda()

	train(model, optimizer, x, y, A, train_mask, val_mask, n_epochs=100, plot=True, device=device)

	# Evaluate on test set
	with torch.no_grad():
		test_mask = data.test_mask.to(device)
		model.eval()
		out = model(x, A)
		test_loss = F.cross_entropy(out[test_mask], y[test_mask])
		test_acc = accuracy(out[test_mask], y[test_mask])

		print(f'---- Accuracy on test set: {test_acc.item()}')
