# Graph Attention Network

A pytorch implementation of the Graph Attention Network (GAT) model presented by [Veličković et. al](https://arxiv.org/abs/1710.10903).
This repo is a quick and somewhat inaccurate reproduction of the method described in the paper. 
The official repository (in Tensorflow) is available in https://github.com/PetarV-/GAT. This repo is inspired by the pytorch implementation of 
Diego Antognini (https://github.com/Diego999/pyGAT).

In the paper they present graph attention networks (GATs), a novel neural network architectures
that operate on graph-structured data, leveraging masked (multi-headed) self-attentional layers to
address the shortcomings of prior methods based on graph convolutions or their
approximations. The figure below displays the multi-headed attention meachanism. 

<img src="https://i.imgur.com/kQEMbXF.png" alt="" width="600"/>



### Evaluation
On the [Cora dataset](https://relational.fit.cvut.cz/dataset/CORA) the model achieved ~81% accuracy on the test set. It uses ~0.35sec per epoch on 2x Nvidia RTX 2080 Ti 11GB.
In the original paper they report reaching an accuracy of 83 ± 0.7%.


On the [CiteSeer dataset](https://linqs.soe.ucsc.edu/data) it achieved ~69% accuracy on the test set. It uses ~0.4sec per epoch on 2x Nvidia RTX 2080 Ti 11GB.
In the original paper they report reaching an accuracy of 72.5 ± 0.7%.


#### Cora
Loss | Accuracy
:--- | :--- 
![](/outputs/Cora/att_loss_plot.png) | ![](/outputs/Cora/att_accuracy_plot.png)


#### CiteSeer
Loss | Accuracy
:--- | :--- 
![](/outputs/CiteSeer/att_loss_plot.png) | ![](/outputs/CiteSeer/att_accuracy_plot.png)
