import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def create_adjacency_matrix(num_nodes, edge_index, add_self_loops=True, normalize=True, device=None):
	"""Creates adjacency matrix from pytorch_geometric edge_index"""
	adj = torch.zeros((num_nodes, num_nodes))
	edges = torch.stack((edge_index[0], edge_index[1]), 1)
	for (source_i, target_i) in edges:
		adj[source_i, target_i] = 1

	if add_self_loops:
		adj = adj + torch.eye(adj.size(0))

	if normalize:
		d = adj.sum(1)
		adj = adj / d.view(-1, 1)

	return adj.to(device)


def print_info_about_dataset(dataset):
	""" Expects dataset from torch_geometric.datasets"""
	try:
		data = dataset.data
		print('------ PRINTING INFO ------')
		print('>> INFO')
		print(f'\tLength of dataset (number of graphs): {len(dataset)}')
		if len(dataset) == 1:
			print(f'\tData is one big graph.')
		else:
			print(f'\tData contains multiple graphs.')

		print(f'\tNum nodes: {data.num_nodes}')
		print(f'\tNum node features: {dataset.num_node_features}')
		print(f'\tNum classes (node or graph classes): {dataset.num_classes}')
		print(f'\tEdges contains attributes: {"False" if data.edge_attr is None else "True"}')
		print(f'\tTarget (y) shape: {data.y.shape}')

		if len(dataset) == 1:
			print(f'\tNode feature matrix shape: {data.x.shape}')
			print(f'\tContains isolated nodes: {data.contains_isolated_nodes()}')
			print(f'\tContains self-loops: {data.contains_self_loops()}')
			print(f'\tEdge_index shape: {data.edge_index.shape}')
			print(f'\tEdges are directed: {data.is_directed()}')
		else:
			print(f'\tPrinting info about first graph:')
			print(f'\t{dataset[0]}')
		print('>> END')

	except:
		print('Some prints failed.')


def plot_dataset(dataset):
	edges_raw = dataset.data.edge_index.numpy()
	edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
	labels = dataset.data.y.numpy()

	G = nx.Graph()
	G.add_nodes_from(list(range(np.max(edges_raw))))
	G.add_edges_from(edges)
	plt.subplot(111)
	options = {
		'node_size': 30,
		'width': 0.2,
	}
	nx.draw(G, with_labels=False, node_color=labels.tolist(), cmap=plt.cm.tab10, font_weight='bold', **options)
	plt.show()


def accuracy(output, labels):
	with torch.no_grad():
		preds = output.max(1)[1].type_as(labels)
		correct = preds.eq(labels).double()
		correct = correct.sum()
		return correct / len(labels)

