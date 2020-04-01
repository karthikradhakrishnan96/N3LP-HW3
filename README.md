# N3LP-HW3
This repository contains our re-implementation of SOTA for HW3.

Implemented by Karthik Radhakrishnan, Sharanya Chakravarthy, Tushar Kanakagiri (Everyone contributed equally)

The sesame street package contains our implementation of BERT using standard Pytorch NN Modules.

Steps to run the code : 
- Download BERT pre-trained model and place it in saved_models directory as per saved_models/instructions.txt
- Unzip data_proc/saved_data_RumEval2019.zip onto data/saved_data_RumEval2019
- Run runner.py 

The system will start printing out the validation F1 scores after every epoch. The score reported by the BUT-FIT paper is 51.4 with BERT-base. After about 60 epochs, it should exceed the score. 

Inspiration and data pre-processing credits - 
https://www.aclweb.org/anthology/S19-2192/
```
@inproceedings{fajcik2019but,
  title={BUT-FIT at SemEval-2019 Task 7: Determining the Rumour Stance with Pre-Trained Deep Bidirectional Transformers},
  author={Fajcik, Martin and Smrz, Pavel and Burget, Lukas},
  booktitle={Proceedings of the 13th International Workshop on Semantic Evaluation},
  pages={1097--1104},
  year={2019}
}
```
