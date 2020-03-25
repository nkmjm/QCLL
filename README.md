# Quantum circuit-like learning

### Available methods
* Quantum circuit-like learning:<br>
Quantum circuit-like learning (QCL) is a classical machine learning algorithm with similar properties, behavior, and performance to quantum circuit learning (QCL). While QCL can employ an exponentially high-dimensional Hilbert space as its feature space due to the use of a quantum circuit [1], QCLL uses the same Hilbert space with a low computational cost by a statistical technique known as count sketch [2]. 

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

### Further reading:
* [Mitarai et al. (2018). "Quantum circuit learning."](https://arxiv.org/abs/1803.00745)
* [Pham & Pagh (2013). "Fast and scalable polynomial kernels via explicit feature maps."](https://chbrown.github.io/kdd-2013-usb/kdd/p239.pdf)
