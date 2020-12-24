# Graph Attention Network

This is a pytorch implementation of the Graph Attention Network (GAT) model presented by [Veličković et. al](https://arxiv.org/abs/1710.10903).

In the paper they present graph attention networks (GATs), a novel neural network architectures
that operate on graph-structured data, leveraging masked multi(-self)-attentional layers to
address the shortcomings of prior methods based on graph convolutions or their
approximations. The figure below displays the multi-headed attention meachanism. 

<img src="https://i.imgur.com/kQEMbXF.png" alt="" width="600"/>

This repo is a quick and in many ways inaccurate reproduction of the method described in the paper. 
The official repository (in Tensorflow) is available in https://github.com/PetarV-/GAT.

### Evaluation
The network is tested on the [Cora dataset](https://relational.fit.cvut.cz/dataset/CORA), and achieved a little over 80% accuracy on test set.


Loss | Accuracy
:--- | :--- 
![](/outputs/att_loss_plot.png) | ![](/outputs/att_accuracy_plot.png)
