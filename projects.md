---
layout: page
title: Projects
permalink: /projects/
---
## Neural Architecture Search on Quantized Neural Nets | _Python_, _PyTorch_
Advisor - [Surekha Bhanot]
- Implemented QNNs from scratch using Pytorch. Wrote Binary layers and Convolutional layers and was able to replicate the paper accuracy **96%** on **MNIST**
- Integrated the above layers with the **Neural Network Intelligence (NNI)** library's ENAS module to perform experiments and judge the accuracy and feasibility of QNNs. Reproduced similar accuracy of **53%** as achieved by a normal Neural network layer
## Sarcasm Detection in Amazon Review Dataset | _Python_, _PyTorch_ | [Github Link](https://github.com/sanchit-ahuja/Sarcasm-Detection-In-Amazon-Reviews)
Advisor - [Surekha Bhanot]
- Built a Sarcasm Detection model from scratch using Pytorch. Achieved an accuracy of **83.45%** for the Sentiment model and accuracy in the range of **55 - 62%** on the **OCEAN** dataset using a CNN model on the word embeddings
- Extracted the features from the above models, concatenated them and used them to train a **CNN-SVM** classifier on the Amazon Review Dataset. Achieved an accuracy of **88%**

## Statistical Machine Translation using IBM Models | _Python_ | [Github Link](https://github.com/sanchit-ahuja/statistical-machine-translation-project)
Advisor - [Lavika Goel]
- Implemented IBM model 1 and model 2 from scratch to translate between Dutch and English
- Optimized the usage of dataset by using random sampling to select around 10,000 samples on which training was performed
- Achieved a cosine score of **0.54** on a randomly sampled test data set

## Registration Software (RegSoft) | _Python, Django, REST, BootStrap, NGINX_
- Built a Registration Software (RegSoft) to handle registration, accommodation and participation of incoming participants for one of **Asia's largest fest** - Oasis.
- Handled the issue of large number of requests using **Signals** in Django. Reduced the load on the webapp by **20%** from which other services were requesting access

[Surekha Bhanot]: https://www.bits-pilani.ac.in/Pilani/surekha/profile
[Lavika Goel]: http://mnit.ac.in/faculty/profile.php?fid=l0DuC-sUKetr0bih-QnNAqSM6AzPscczKXosr543BMg