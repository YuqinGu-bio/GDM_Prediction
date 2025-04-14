import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score,recall_score
import matplotlib.pyplot as plt
import shap
import pickle
#train data
df=pd.read_csv('./input/phenotype_qtrans_train_data_PRSicesnpPRS_plink_dosage_2w_20week_0.6_1501_6979_241220.txt',sep='\t')
df=df.iloc[:,1:]
print(df['GDM'].value_counts())
df=df.dropna(axis='index',how='all',subset=['GDM'])
print(df.head())
df['GDM']=df['GDM']-1
print(df.head())
print(df['GDM'].value_counts())
x_1,x_2,x_3,y_1=df.iloc[:,1:],df.iloc[:,1:47],df.iloc[:,47:],df.iloc[:,0]
print(x_2.head())
print(x_3.head())
#read test data
x_data=[x_1,x_2,x_3]
#ramdom split 5 fold data
for i in range(1,11):
    model_index = 1
    for independent_variable in x_data:
        dtrain=xgb.DMatrix(independent_variable, label=y_1)
        def objective(trial):
            param={
                'objective':"binary:logistic",
                'eval_metric':"auc",
                'max_depth':trial.suggest_int('max_depth', 2, 10),
                'learning_rate':trial.suggest_loguniform('learning_rate', 1e-5, 1),
                'num_boost_round':trial.suggest_int('num_boost_round', 10, 200)
            }
            cv_results=xgb.cv(params=param,
                              dtrain=dtrain,
                              early_stopping_rounds=10,
                              nfold=5,
                              metrics='auc',
                              stratified=True)
            return cv_results['test-auc-mean'].iloc[-1]
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        #get best params
        best_xgb_params = study.best_params
        print(best_xgb_params)
        #Train model with best parameters
        best_model = xgb.train(
            params=best_xgb_params,
            dtrain=dtrain
        )
        #save model
        best_model.save_model('./output/best_xgb_model.json_'+str(i) + '_' + str(model_index))
        #load model
        loaded_model = xgb.Booster()
        loaded_model.load_model('./output/best_xgb_model.json_'+str(i) + '_' + str(model_index))
        #predict case prob
        dtrain_preds = loaded_model.predict(dtrain)
        train_preds_df = pd.DataFrame({
            'Sample_ID': range(len(dtrain_preds)),
            'GDM':y_1,
            'Predicted_Prob': dtrain_preds,
            'rep_index':i,
            'model_index':model_index
        })
        #save prob in csv file
        train_preds_df.to_csv('./output/train_predictions.csv', index=False,mode='a')
        #prob convert to binary
        train_preds_binary = (dtrain_preds > 0.5).astype(int)
        train_accuracy = accuracy_score(y_1, train_preds_binary)
        train_precision = precision_score(y_1, train_preds_binary)
        train_recall = recall_score(y_1, train_preds_binary)
        train_f1 = f1_score(y_1, train_preds_binary)
        train_auc = roc_auc_score(y_1, dtrain_preds)
        rep_index=i
        metrics = [[train_accuracy,train_precision,train_recall,train_f1,train_auc,rep_index,model_index]]
        #construct DataFrame
        metrics_df = pd.DataFrame(metrics,columns=['accuracy_score', 'precision_score', 'recall_score', 'f1_score',
                                             'roc_auc','rep_index', 'model_index'])
        #save to CSV file
        metrics_df.to_csv('./output/model_metrics.csv', index=False,mode='a')
        #shap
        explainer = shap.TreeExplainer(loaded_model)
        shap_values = explainer.shap_values(independent_variable)
        #plot
        shap.summary_plot(shap_values,independent_variable, show=False)
        plt.rcParams['font.size'] = 12
        plt.savefig('./output/impact_PRS_pheno_all_' + str(
                i) + '_' + str(model_index) + '.png', dpi=300)
        plt.close()
        shap.summary_plot(shap_values,independent_variable, plot_type="bar")
        plt.savefig(
            './output/barplot_PRS_pheno_all_' + str(
                i) +'_' + str(model_index) + '.png', dpi=300)
        plt.close()
        #save feature importance data
        feature_importance = pd.DataFrame()
        feature_importance['feature'] = independent_variable.columns
        feature_importance['importance'] = np.abs(shap_values).mean(0)
        feature_importance.sort_values('importance', ascending=False)
        feature_importance['rep_index'] = i
        feature_importance['model_index'] = model_index
        feature_importance.to_csv('./output/feature_importance.csv',
            mode='a',index=False)
        model_index = model_index + 1






