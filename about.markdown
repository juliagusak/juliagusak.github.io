---
layout: page
title: About
permalink: /about/
excerpt: Academic web page
does_not_need_title: true
---
<h1 style="margin:0px; font-size: 36px">About</h1>

- Currently, I am a Machine Learning researcher at <a href="https://faculty.skoltech.ru/people/juliagusak">Skoltech</a> at <a href="http://www.deeptensor.ml/team.html">Tensor networks and deep learning for applications in data mining</a> laboratory, working with <a href="https://scholar.google.com/citations?user=5kMqBQEAAAAJ&hl=en">Prof. Ivan Oseledets</a> and <a href="https://scholar.google.com/citations?user=wpZDx1cAAAAJ&hl=en">Prof. Andrzej Cichocki</a>.

- I received my <a href="http://mech.math.msu.su/~snark/files/diss/0169diss.pdf">Ph.D. in Probability Theory and Statistics</a> from <a href="http://new.math.msu.su/department/probab/os/index-e.html">Lomonosov Moscow State University</a>, and in parallel, I completed a Master's-level programm in Computer Science and Data Analysis from the <a href="https://yandexdataschool.com/">Yandex School of data analysis</a>.

- My recent research deals with compression and acceleration of computer vision models (classification/object detection/segmentation),  as well as neural networks analysis using low-rank methods, such as tensor decompositions and active subspaces.  Also, I have some audio-related activity,  particularly, I participate in the project on speech synthesis and voice conversion. Some of my earlier projects were related to medical data processing (EEG, ECG)  and included human disease detection, artifact removal, and weariness detection.

- Research interests: deep learning (DL), interpretability of DL, computer vision, speech technologies, multi-modal/multi-task learning, semi-supervised/unsupervised learning, transfer learning, domain adaptation, hyper networks, tensor decompositions for DL.


<br/>
<div class="scaleIcons">
<center>
    <a class="hovernounderline" href="https://docs.google.com/document/d/1ykxd4l7kpEF0gWiyk-msFoLbVSlY-mBO2yond_nfWto/edit?usp=sharing">
        <i class="svg-icon cv"></i>
    </a>
    <a class="hovernounderline" href="https://github.com/{{ site.footer-links.github }}">
        <i class="svg-icon github"></i>
    </a>
    <a class="hovernounderline" href="https://scholar.google.com/citations?hl=en&user={{ site.footer-links.googlescholar }}">
        <i class="svg-icon googlescholar"></i>
    </a>
    <a class="hovernounderline" href="https://www.linkedin.com/in/{{ site.footer-links.linkedin }}">
        <i class="svg-icon linkedin"></i>
    </a>
    <a class="hovernounderline" href="https://www.twitter.com/{{ site.footer-links.twitter }}">
        <i class="svg-icon twitter"></i>
    </a>
</center>
</div>

<h1 style="font-size: 36px">Selected publications</h1>
<!-- <> -->
<div class="row publications">
    <div class="col-sm-5 vcenter marginbottom">
        <img class="img-responsive pub-image" src="/assets/about/cpd-epc.png" alt="" />
    </div>
    <div class="col-sm-7 vcenter" style="margin-right: -4px; text-align: justify;">
        <p class="title">Stable Low-rank Tensor Decomposition for Compression of Convolutional Neural Network</p>
        <p class="authors">A.H. Phan, K. Sobolev, K. Sozykin, D. Ermilov, J. Gusak, P. Tichavsky, V. Glukhov, I. Oseledets, A. Cichocki</p>
        <p class="conf">ECCV 2020</p>
        <p class="description">
            Most state of the art deep neural networks are overparameterized and exhibit a high computational cost. A straightforward approach to this problem is to replace convolutional kernels with its low-rank tensor approximations, whereas the Canonical Polyadic tensor Decomposition is one of the most suited models. However, fitting the convolutional tensors by numerical optimization algorithms often encounters diverging components, i.e., extremely large rank-one tensors but canceling each other. Such degeneracy often causes the non-interpretable result and numerical instability for the neural network fine-tuning. This paper is the first study on degeneracy in the tensor decomposition of convolutional kernels. We present a novel method, which can stabilize the low-rank approximation of convolutional kernels and ensure efficient compression while preserving the high-quality performance of the neural networks. We evaluate our approach on popular CNN architectures for image classification and show that our method results in much lower accuracy degradation and provides consistent performance.
        </p>
        <div class="links">
            <a href="https://arxiv.org/pdf/2008.05441.pdf">Paper</a>
        </div>
    </div>
</div>

<div class="row publications">
    <div class="col-sm-5 vcenter marginbottom">
        <img class="img-responsive pub-image" src="/assets/about/musco.png" alt=""/>
    </div>
    <div class="col-sm-7 vcenter" style="margin-right: -4px; text-align: justify;">
        <p class="title">Automated Multi-Stage Compression of Neural Networks</p>
        <p class="authors"> <b>J. Gusak, M. Kholiavchenko</b>, E. Ponomarev, L. Markeeva, A. Cichocki, I. Oseledets</p>
        <p class="conf">ICCV 2019 Workshop on Low-Power Computer Vision</p>
        <p class="description">
            We propose a new simple and efficient iterative approach for compression of deep neural networks, which alternates low-rank factorization with smart rank selection and fine-tuning. We demonstrate the efficiency of our method comparing to non-iterative ones. Our approach improves the compression rate while maintaining the accuracy for a variety of computer vision tasks.
        </p>
        <div class="links">
            <a href="http://openaccess.thecvf.com/content_ICCVW_2019/papers/LPCV/Gusak_Automated_Multi-Stage_Compression_of_Neural_Networks_ICCVW_2019_paper.pdf">Paper</a>
            <a href="https://rebootingcomputing.ieee.org/images/files/pdf/iccv-2019_julia-gusak.pdf">Presentation</a>
            <a href="https://github.com/musco-ai/musco-pytorch">Code-PyTorch</a>
            <a href="https://github.com/musco-ai/musco-tf">Code-TensorFlow</a>
        </div>
    </div>
</div>

<div class="row publications">
    <div class="col-sm-5 vcenter marginbottom">
        <img class="img-responsive pub-image" src="/assets/about/neural-ode-norm.png" alt=""/>
    </div>
    <div class="col-sm-7 vcenter" style="margin-right: -4px; text-align: justify;">
        <p class="title">Towards Understanding Normalization in Neural ODEs</p>
        <p class="authors"> <b>J. Gusak</b>, L. Markeeva, T. Daulbaev, A. Katrutsa, A. Cichocki, I. Oseledets</p>
        <p class="conf">ICLR 2020 DeepDiffeq workshop</p>
        <p class="description">
            Normalization is an important and vastly investigated technique in deep learning. However, its role for Ordinary Differential Equation based networks (neural ODEs) is still poorly understood. This paper investigates how different normalization techniques affect the performance of neural ODEs. Particularly, we show that it is possible to achieve 93% accuracy in the CIFAR-10 classification task, and to the best of our knowledge, this is the highest reported accuracy among neural ODEs tested on this problem.
        </p>
        <div class="links">
            <a href="https://openreview.net/forum?id=mllQ3QNNr9d">Paper</a>
            <a href="https://drive.google.com/file/d/1rbpudtm01WfpYPJaew-OmTK1BaDwQE0p/view?usp=sharing">Presentation</a>
            <a href="https://github.com/juliagusak/neural-ode-norm">Code</a>
        </div>
    </div>
</div>

<div class="row publications">
    <div class="col-sm-5 vcenter marginbottom">
        <img class="img-responsive pub-image" src="/assets/about/iam.png" alt=""/>
    </div>
    <div class="col-sm-7 vcenter" style="margin-right: -4px; text-align: justify;">
        <p class="title">Interpolation technique to speed up gradients propagation in neural ordinary differential equations</p>
        <p class="authors"> T. Daulbaev, A. Katrutsa, L. Markeeva, J. Gusak, A. Cichocki, I. Oseledets</p>
        <p class="conf">arxiv 2020</p>
        <p class="description"> We propose a simple interpolation-based method for the efficient approximation of gradients in neural ODE models. We compare it with reverse dynamic method (known in literature as ''adjoint method'') to train neural ODEs on classification, density estimation and inference approximation tasks. We also propose a theoretical justification of our approach using logarithmic norm formalism. As a result, our method allows  faster model training than reverse dynamic method on several standard benchmarks.
        </p>
        <div class="links">
            <a href="https://arxiv.org/abs/2003.05271">Paper</a>
        </div>
    </div>
</div>

<div class="row publications">
    <div class="col-sm-5 vcenter marginbottom">
        <img class="img-responsive pub-image" src="/assets/about/ron.png" alt=""/>
    </div>
    <div class="col-sm-7 vcenter" style="margin-right: -4px; text-align: justify;">
        <p class="title">Reduced-Order Modeling of Deep Neural Networks</p>
        <p class="authors"><b>J. Gusak, T. Daulbaev</b> E. Ponomarev, A. Cichocki, I. Oseledets</p>
        <p class="conf">arxiv 2019 (accepted to Computational Mathematics and Mathematical Physics Journal)</p>
        <p class="description">
            We introduce a new method for speeding up the inference of deep neural networks. It is somewhat inspired by the reduced-order modeling techniques for dynamical systems. The cornerstone of the proposed method is the maximum volume algorithm. We demonstrate efficiency on VGG and ResNet architectures pre-trained on different datasets. We show that in many practical cases it is possible to replace convolutional layers with much smaller fully-connected layers with a relatively small drop in accuracy.
        </p>
        <div class="links">
            <a href="https://arxiv.org/pdf/1910.06995">Paper</a>
        </div>
    </div>
</div>

<div class="row publications">
    <div class="col-sm-5 vcenter marginbottom">
        <img class="img-responsive pub-image" src="/assets/about/asnet.png" alt=""/>
    </div>
    <div class="col-sm-7 vcenter" style="margin-right: -4px; text-align: justify;">
        <p class="title">Active Subspace of Neural Networks: Structural Analysis and Universal Attacks</p>
        <p class="authors"> C. Cui, K. Zhang, T. Daulbaev, J. Gusak, I. Oseledets, Z. Zhang</p>
        <p class="conf">arxiv 2019 (accepted to SIAM Journal on Mathematics of Data Science, SIMODS)</p>
        <p class="description">
            Active subspace is a model reduction method widely used in the uncertainty quantification community. Firstly, we employ the active subspace to measure the number of" active neurons" at each intermediate layer and reduce the number of neurons from several thousands to several dozens, yielding to a new compact network. Secondly, we propose analyzing the vulnerability of a neural network using active subspace and finding an additive universal adversarial attack vector that can misclassify a dataset with a high probability.
        </p>
        <div class="links">
            <a href="https://arxiv.org/pdf/1910.13025.pdf">Paper</a>
            <a href="https://github.com/chunfengc/ASNet">Code</a>
        </div>
    </div>
</div>


<br/>
<h1 style="font-size: 36px">Selected projects</h1>
<!-- < -->
<!-- <div id="projects"> -->
<div class="row projects">
    <div class="col col-sm-3 vcenter imgcol marginbottom">
        <img class="img-responsive proj-img" src="/assets/about/dataloaders.jpg" alt=""/>
    </div>
    <div class="col col-sm-9 vcenter" style="margin-right: -4px; text-align: justify;">
        <p class="title">Data loaders for speech and audio data sets</p>
        <p class="description">
        A Python library with PyTorch and TFRecords data loaders for convenient preprocessing of popular speech, music and environmental sound data sets.
        </p>
        <div class="links">
            <a href="https://github.com/juliagusak/dataloaders">Code</a>
        </div>
    </div>
</div>

<div class="row projects">
    <div class="col col-sm-3 vcenter imgcol marginbottom">
        <img class="img-responsive proj-img" src="/assets/about/flopco.jpg" alt=""/>
    </div>
    <div class="col col-sm-9 vcenter" style="margin-right: -4px; text-align: justify;">
        <p class="title">FlopCo: FLOP and other statistics COunter for Pytorch neural networks</p>
        <p class="description">
        A Python library FlopCo has been created to make FLOP and MAC counting simple and accessible for  Pytorch  neural networks. Moreover, FlopCo allows to collect other useful model statistics, such as number of parameters, shapes of layer inputs/outputs, etc.
        </p>
        <div class="links">
            <a href="https://github.com/juliagusak/flopco-pytorch">Code</a>
        </div>
    </div>
</div>

<div class="row projects">
    <div class="col col-sm-3 vcenter imgcol marginbottom">
        <img class="img-responsive proj-img" src="/assets/about/audio_classification.jpg" alt=""/>
    </div>
    <div class="col col-sm-9 vcenter" style="margin-right: -4px; text-align: justify;">
        <p class="title">Speaker identification using neural network models</p>
        <p class="description">
        In the framework of this project a <a href="https://arxiv.org/abs/1704.01279">WaveNet-style autoencoder</a> model for audio synthesis and a <a href="https://arxiv.org/abs/1711.10282">neural network</a> for environment sounds classification have been adopted to solve speaker identification task. 
        </p>
        <div class="links">
            <a href="https://github.com/juliagusak/audio_classification">Code</a>
        </div>
    </div>
</div>

<div class="row projects">
    <div class="col col-sm-3 vcenter imgcol marginbottom">
        <img class="img-responsive proj-img" src="/assets/about/scatnet.png" alt=""/>
    </div>
    <div class="col col-sm-9 vcenter" style="margin-right: -4px; text-align: justify;">
        <p class="title">Scattering transform</p>
        <p class="description">
         Python implementation of the scattering transform from MATLAB toolbox <a href="https://www.di.ens.fr/data/software/scatnet">ScatNet</a>. 
        </p>
        <div class="links">
            <a href="https://github.com/juliagusak/scatnet_python">Code</a>
        </div>
    </div>
</div>

<div class="row projects">
    <div class="col col-sm-3 vcenter imgcol marginbottom">
        <img class="img-responsive proj-img" src="/assets/about/ftrl-proximal.png" alt=""/>
    </div>
    <div class="col col-sm-9 vcenter" style="margin-right: -4px; text-align: justify;">
        <p class="title">Per-Coordinate FTRL-Proximal with L1 and L2 Regularization for Logistic Regression</p>
        <p class="description">
         C++ implementation of the online optimization algorithm for logistic regression training, described in the following <a href="http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf">paper.</a> 
        </p>
        <div class="links">
            <a href="https://github.com/juliagusak/ftrl-proximal">Code</a>
        </div>
    </div>
</div>

