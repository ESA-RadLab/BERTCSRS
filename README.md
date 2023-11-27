# BERT for Complex Systematic Review Screening (BERTCSRS)

## Model architecture & training
The model consists of a variable version of [BERT](https://huggingface.co/blog/bert-101) with a fully-connected 
classification layer on top, feeding into a single binary classification node. The BERT outputs and the classification 
layer have a ReLU activation layer and a dropout of 0.2 during training. The output node has a sigmoid activation function, 
which can be turned off for training, as I use BCEWithLogitsLoss(), which applies a sigmoid layer itself. BCEWithLogitsLoss() 
is implemented with a strong weight on positive samples, to emphasize recall. The optimizer used is RAdam() and I 
implemented early stopping with a timeout of 3 epochs. See [this](Project%20info.pptx) for general info on the project.

## Note on the remote machine
To prevent losing all training progress when the ssh connection times out during a long-running process, I use the `screen` 
utility which will preserve any processes running in it when the connection drops. 
[Here is a link to some useful examples of how to use screen](https://www.tecmint.com/screen-command-examples-to-manage-linux-terminals/). 
A downside of using screen is that scroll doesn't work, so I don't use it when evaluating the model training.

## Workflows
These examples use 'CNS' as the data label and 'pubmed_abstract' as the bert model
### Simple training and testing
1\. run [split_train_test.py](data/split_train_test.py) on raw training data

2\. push `data/CNS` to git

3\. edit [train.py](train.py)'s \_\_main__ method and push to git  

**\~VM\~** `cd BERTCSRS; conda activate snakes`  
4. `git pull`  
5. `python train.py`  
6. `git add output/logs; git commit -m "logs pubmed_abstract CNS"; git push`  

**\~Local\~**  
7. pull from git  

Model logs are in `output/logs/pubmed_abstract/{version datetime}`  
Model state dict stays on the VM in `output/models/pubmed_abstract/{version datetime}`  
The state dict is too large for git, but can be retrieved using scp:  
`***REMOVED***`

### Kfold analysis
1\. run [split_kfold.py](data/split_kfold.py) on raw training data  
2\. push `Kfolds/data/CNS/` to git (4 files per fold)  
3\. edit [Kfold.py](Kfold.py) and push to git  

**\~VM\~** `cd BERTCSRS; conda activate snakes; screen -S bert`  
4. `git pull`  
5. `python Kfold.py`  
6. `git add Kfolds; git commit -m "output kfold CNS"; git push`   

**\~Local\~**  
7. pull from git  

The result summary is in `Kfolds/Kfold_results_CNS.xlsx` in the tab 'pubmed_abstract'  
Test outputs of the best epoch of every fold are in `Kfolds/output/CNS/{start datetime}`  
The data used is in `Kfolds/data/CNS/`  
Training logs and model state dict stay on the VM in `output/models/pubmed_abstract/{version datetime}`

### Classification
1\. run [data_prep.py](data/data_prep.py) on raw source data  

2\. push `data/processed_datasets/all_ref_CNS.csv` to git  

3\. edit [run_classification.py](classification/run_classification.py)'s \_\_main__ method and push to git  

**\~VM\~** `cd BERTCSRS; conda activate snakes; screen -S bert`  
4. `git pull`  
5. `python -m classification.run_classification`  
6. `git add data/output; git commit -m "output classification CNS"; git push`  

**\~Local\~**  
7. run [test.py](evaluation/test.py) on val data (can also be done on VM, or re-split the test output)  
8. run [evaluate_output.py](evaluation/evaluate_output.py) on test output  
9. choose a threshold  
10. validate threshold on val data with [threshold_output.py](evaluation/threshold_output.py)  
11. git pull classification output from VM  
12. run [classify_output.py](classification/classify_output.py) on the output in `data/output/FULL/CNS`, 
using chosen threshold  
    (use line 31 to set a second threshold to get predictions close to the chosen threshold in an Excel file)  
13. run [classified_to_syrf.py](classification/classified_to_syrf.py) when you are satisfied with the results.   
        (This code is greatly adapted to the peculiarities of the source files used here, be aware of this when using it for a different file)

The output is in `data/output/FULL/CNS`, `CNS_..._DECISION.csv` contains all abstracts, the model prediction and decision, 
`CNS_..._INCLUDED.csv` contains only the titleabstracts included by the model, and `CNS_..._CHECK.xlsx` can contain abstracts 
below the set threshold but above a second one, to evaluate the predictions between these two thresholds
The .csv file that can be uploaded to SYRF is in `data/output`

