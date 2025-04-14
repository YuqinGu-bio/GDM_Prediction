The methods for analyzing the NIPT data for GWAS have been posted on our githup(https://github.com/liusylab/NIPT-human-genetics). 

The detail information of prediction model was showed as below，the results of this study can be reproduced using the code and data in the following files.

Require software: Python 3
Code file name: prediction_initial.py and prediction_test.py

Input data: 
Train dataset: phenotype_qtrans_train_data_PRSicesnpPRS_plink_dosage_2w_20week_0.6_1501_6979.txt
Test dataset:phenotype_qtrans_test_data_PRSicesnpPRS_plink_dosage_2w_20week_0.2_550_2276.txt
 
A round of load model about 1 minutes.

Expected output:
The probability of GDM file:test_predictions.csv
Sample_ID,GDM,Predicted_Prob,rep_index,model_index
0,0,0.12506932,1,1
1,0,0.23683657,1,1
2,0,0.15557861,1,1
3,1,0.14000589,1,1
4,0,0.031583734,1,1

The model performance file:test_model_metrics.csv
accuracy_score,precision_score,recall_score,f1_score,roc_auc,rep_index,model_index
0.835102618542109,0.9565217391304348,0.16,0.27414330218068533,0.7285428982265537,1,1

False positive rate and true positive rate file:model_rep_index_model_index.csv
Feature importance ofmodel:test_feature_importance.csv
feature,importance,rep_index,model_index
Age,0.022820575,1,1
BMI,0.02115543,1,1
SBP,0.0,1,1
DBP,0.003077178,1,1
HbF,0.0,1,1

model_index：the symbol of prediction model,1 is combined model,2 is only clinical phenotype model and 3 is the PRS model.
rep_index: the symbol of external repeat order.