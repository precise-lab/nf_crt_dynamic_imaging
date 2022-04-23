# nf_crt_dynamic_imaging
Neural Fields for solving dynamic CRT imaging problems

Neural fields are a particular class of neural networks that represent the dynamic object as a continuous function of space and time. Neural field representation reduces image reconstruction to estimating the network parameters via a nonlinear optimization problem (training). Once trained, the neural field can be evaluated at arbitrary locations in space and time, allowing for high-resolution rendering of the object. Key advantages of the proposed approach are that neural fields automatically learn and exploit redundancies in the sought-after object to both regularize the reconstruction and significantly reduce memory storage requirements. 

In this repository we display this proposed neural field framework with a supervised learning example and two unsupervised image reconstruction examples using the dynamic circular radon transform (CRT).

Please cite our upcoming journal article “A Memory-Efficient Dynamic Image Reconstruction Method using Neural Fields”.
