[![DOI](https://zenodo.org/badge/484778102.svg)](https://zenodo.org/badge/latestdoi/484778102)

# Neural Field CRT Dynamic Imaging

Companion code of journal articles

> L. Lozenski, M. Anastasio, U. Villa. _A Memory-Efficient Self-Supervised Dynamic Image Reconstruction Method using Neural Fields_, IEEE Transactions on Computational Imaging 8 (2022): 879-892. ([preprint](https://arxiv.org/abs/2205.05585?context=eess))
> L. Lozenski, R. Cam, M. Pagel, M. Anastasio. _ProxNF: Neural Field Proximal Training for High-Resolution 4D Dynamic Image Reconstruction_, submitted to IEEE Transactions on Computational Imaging. ([preprint](https://arxiv.org/abs/2403.03860))



Neural Fields for solving dynamic CRT imaging problems

Neural fields are a particular class of neural networks representing the dynamic object as a continuous function of space and time. Neural field representation reduces image reconstruction to estimate the network parameters via a nonlinear optimization problem (training). Once trained, the neural field can be evaluated at arbitrary locations in space and time, allowing for high-resolution rendering of the object. Key advantages of the proposed approach are that neural fields automatically learn and exploit redundancies in the sought-after object to both regularize the reconstruction and significantly reduce memory storage requirements. 

In this repository, we display this proposed neural field framework with a supervised learning example and two unsupervised image reconstruction examples using the dynamic circular radon transform (CRT).


# Dependencies 

`PyTorch`: open source machine learning framework that accelerates the path from research prototyping to production deployment.
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

`scikit-image`: collection of algorithms for image processing
```bash
conda install scikit-image
```
