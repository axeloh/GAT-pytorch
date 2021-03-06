# Graph Attention Network

A pytorch implementation of the Graph Attention Network (GAT) model presented by [Veličković et. al](https://arxiv.org/abs/1710.10903).
This repo is a quick and somewhat inaccurate reproduction of the method described in the paper. 
The official repository (in Tensorflow) is available in https://github.com/PetarV-/GAT. This repo is inspired by the pytorch implementation made by 
Diego Antognini (https://github.com/Diego999/pyGAT).

In the paper they present graph attention networks (GATs), a novel neural network architectures
that operate on graph-structured data, leveraging masked (multi-headed) self-attentional layers to
address the shortcomings of prior methods based on graph convolutions or their
approximations. The figure and formulas below are adapted from the paper. 

The aggregation process of a multi-head graph attentional layer:

<img src="https://i.imgur.com/kQEMbXF.png" alt="" width="600"/>

The output of each head is concatinated:

<img src="https://i.imgur.com/EQqV4Fw.png" alt="" width="260"/>

If multi-head attention is used inn the final layer, averaging is used instead of concatination, and the activation function delayed:

<img src="https://i.imgur.com/OBL4FER.png" alt="" width="300"/>



### Evaluation
I evaluate the model on two standard citation network benchmark datasets: [Cora dataset](https://relational.fit.cvut.cz/dataset/CORA) and [CiteSeer dataset](https://linqs.soe.ucsc.edu/data). I follow the same experimental setup as in the paper. Only 20 nodes per class are used for training. The trained model is evaluated on 1000 test nodes, while 500 nodes are used for validation purposes (mainly early-stopping). The reported results are averaged over 5 runs. 

On Cora the model achieved 80.8 ± 0.5% accuracy on the test set. It uses ~0.35sec per epoch on 2x Nvidia RTX 2080 Ti 11GB.
In the original paper they report reaching an accuracy of 83 ± 0.7% (averaged over 100 runs).


On CiteSeer it achieved 68 ± 0.4% accuracy on the test set. It uses ~0.4sec per epoch on 2x Nvidia RTX 2080 Ti 11GB.
In the original paper they report reaching an accuracy of 72.5 ± 0.7% (averaged over 100 runs).

#### Cora
Loss | Accuracy
:--- | :--- 
![](/outputs/Cora/att_loss_plot.png) | ![](/outputs/Cora/att_accuracy_plot.png)


#### CiteSeer
Loss | Accuracy
:--- | :--- 
![](/outputs/CiteSeer/att_loss_plot.png) | ![](/outputs/CiteSeer/att_accuracy_plot.png)
