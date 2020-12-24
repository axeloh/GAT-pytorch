# Graph Attention Network

This is a pytorch implementation of the Graph Attention Network (GAT) model presented by Veličković et. al (2017, https://arxiv.org/abs/1710.10903).

In the paper they present graph attention networks (GATs), a novel neural network architectures
that operate on graph-structured data, leveraging masked multi(-self)-attentional layers to
address the shortcomings of prior methods based on graph convolutions or their
approximations. The figure below displays the multi-headed attention meachanism. 

<img src="https://i.imgur.com/kQEMbXF.png" alt="" width="600"/>

The official repository for the GAT (Tensorflow) is available in https://github.com/PetarV-/GAT.

### Evaluation
The network is tested on the [Cora dataset](https://relational.fit.cvut.cz/dataset/CORA), and achieved a little over 80% accuracy on test set.
