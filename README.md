# Quantum circuit-like learning

Quantum circuit-like learning (QCLL) is a classical machine learning algorithm with similar properties, behavior, and performance to quantum circuit learning (QCL). While QCL can employ an exponentially high-dimensional Hilbert space as its feature space due to the use of a quantum circuit [1], QCLL uses the same Hilbert space with a low computational cost by a statistical technique known as count sketch [2]. <br><br>
[1] [Mitarai et al. (2018). "Quantum circuit learning."](https://arxiv.org/abs/1803.00745)<br>
[2] [Pham & Pagh (2013). "Fast and scalable polynomial kernels via explicit feature maps."](https://chbrown.github.io/kdd-2013-usb/kdd/p239.pdf)

### Citation
If you use QCLL in your published work, please cite the following preprint :<br>
* Koide-Majima, N., Majima, K. (2020).<br>
“Quantum circuit-like learning: A fast and scalable classical machine-learning algorithm with similar performance to quantum circuit learning”<br>
URL: [https://arxiv.org/abs/2003.10667](https://arxiv.org/abs/2003.10667)<br>

### Requirements
* python 3
* numpy, spicy, random, math, matplotlib (for demo)

### Demo
* demo_regression.ipynb (QCLL on 4 regression tasks; see figure below)<br>
* demo_classification.ipynb (QCLL on a classification task)<br>

The initial state of the optimization is randomly defined. In our preprint, to avoid local minima of the cost function, we repeated the optimization algorithm several times with different initializations and the parameters showing the lowest cost function value were adopted. Please note that you run the optimization algorithm only once for quick results in this demo.

<img src="https://user-images.githubusercontent.com/52347843/77525781-e99d3800-6ecc-11ea-9f8b-760772528f42.jpg" width="600px">
