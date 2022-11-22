# ReNAP: Relation Network with Adaptive Prototypical Learning for Few-Shot Classification

This repo contains the reference source code for the paper [ReNAP: Relation Network with Adaptive Prototypical Learning for Few-Shot Classification](#) in Neurocomputing.

## Enviroment

 - Python=3.7
 - Pytorch=1.7
 - json

## Datasets

* Change directory to `./filelists/CUB`
* run `source ./download_CUB.sh`

## Train & test

Run the train, save_features and test script in directory `./comm`


## References

Our testbed builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

* Framework, Backbone, Compared methods
  https://github.com/wyharveychen/CloserLookFewShot 

## Citation

If you find our code useful, please consider citing our work using the bibtex:

```
@inproceedings{
li2022renap,
title={ReNAP: Relation Network with Adaptive Prototypical Learning for Few-Shot Classification},
author={Xiaoxu Li,Yalan Li,Yixiao Zheng,Rui Zhu,Zhanyu Ma,Jing-Hao Xue,Jie Cao},
booktitle={Neurocomputing},
year={2022}
}
```
