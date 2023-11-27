# BERT for Complex Systematic Review Screening (BERTCSRS)

## Workflows
These examples use 'CNS' as the data label and 'pubmed_abstract' as the bert model
### Simple training and testing
1. run [data/split_train_test.py](data/split_train_test.py) on raw training data
2. push `data/CNS` to git
3. edit [train.py](train.py)'s \_\_main__ method and push to git  

**~VM~**  
`cd BERTCSRS; conda activate snakes; screen -S bert`
4. `git pull`
5. `python train.py`
6. `git add output/logs; git commit -m "logs pubmed_abstract CNS"; git push`  

**~Local~**
7. pull from git  

Model logs are in `output/logs/pubmed_abstract/{version datetime}`  
Model state dict stays on the VM in `output/models/pubmed_abstract/{version datetime}`  
The state dict is too large for git, but can be retrieved using scp:  
`***REMOVED***`

8. 

### Kfold analysis
1. run [data/split_kfold.py](data/split_kfold.py) on raw training data
2. push `Kfolds/data/CNS/` to git (4 files per fold)
3. edit [Kfold.py](Kfold.py) and push to git  

**~VM~**  
`cd BERTCSRS; conda activate snakes; screen -S bert`
4. `git pull`
5. `python Kfold.py`
6. `git add Kfolds; git commit -m "output kfold CNS"; git push`  

**~Local~**
7. pull from git  

Result summary is in `Kfolds/Kfold_results_CNS.xlsx` in the tab 'pubmed_abstract'  
Test outputs of the best epoch of every fold are in `Kfolds/output/CNS/{start datetime}`  
The data used is in `Kfolds/data/CNS/`  
Training logs and model state dict stay on the VM in `output/models/pubmed_abstract/{version datetime}`

### Classification
1. run [data/data_prep.py](data/data_prep.py) on raw source data
2. push `data/processed_datasets/all_ref_CNS.csv` to git
3. edit [classification/run_classification.py](classification/run_classification.py)'s \_\_main__ method and push to git

**~VM~**  
`cd BERTCSRS; conda activate snakes; screen -S bert`
4. `git pull`
5. `python -m classification.run_classification`
6. `git add data/output; git commit -m "output classification CNS"; git push`

**~Local~**  
7. run [evaluation/test.py](evaluation/test.py) on val data (can also be done on VM, or re-split the test output)
8. run [evaluation/evaluate_output.py](evaluation/evaluate_output.py) on test output
9. choose a threshold
10. validate threshold on val data with [evaluation/threshold_output.py](evaluation/threshold_output.py)
11. git pull classification output from VM
12. run [classification/classify_output.py](classification/classify_output.py) on the output in `data/output/FULL/CNS`, 
using chosen threshold  
    (use line 31 to get an Excel file with abstracts close to the chosen threshold)

The output is in `data/output/FULL/CNS`, `CNS_DECISION.csv` contains all abstracts, the model prediction and decision, 
`CNS_INCLUDED.csv` contains only the titleabstracts included by the model, and `CNS_CHECK.xlsx` can contain abstracts 
below the set threshold but above a second one, to evaluate the predictions between these two thresholds  

13. run [classification/classified_to_syrf.py](classification/classified_to_syrf.py) on `CNS_INCLUDED.csv`

