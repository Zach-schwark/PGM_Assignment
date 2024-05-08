from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from  pgmpy.inference.EliminationOrder import WeightedMinFill
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_score, recall_score, roc_auc_score, balanced_accuracy_score

class ModelEvaluation:
    
    
    def performInference(model: BayesianNetwork, testing_evidence: list, testing_targets: pd.Series):
        
        for i in range(len(testing_evidence)):
            for attribute in list(testing_evidence[i].keys()):
                if attribute not in model.nodes():
                    del testing_evidence[i][attribute]
            

        inference = VariableElimination(model)
        elimination_order = WeightedMinFill(model).get_elimination_order(model.nodes(),show_progress=False)
        elimination_order.remove('class')


        y_pred = []
        y_true = []
        for i in range(len(testing_evidence)):
            bunkrupt_or_not = inference.map_query(['class'], evidence=testing_evidence[i],show_progress=False)
            y_pred.append(bunkrupt_or_not['class'])
            y_true.append(testing_targets.iloc[i])
        
        return y_pred, y_true
    
    def evaluate(y_pred:list, y_true: list):
        
        accuracyScore = accuracy_score(y_true, y_pred)
        f1score = f1_score(y_true, y_pred)
        precisionScore = precision_score(y_true, y_pred)
        recallScore = recall_score(y_true, y_pred)
        rocaucscore =  roc_auc_score(y_true, y_pred)
        balancedAccuracyScore = balanced_accuracy_score(y_true, y_pred,adjusted=True)

        f1_score_macro = f1_score(y_true, y_pred, average='macro')
        precision_score_macro = precision_score(y_true, y_pred, average='macro')
        recall_score_macro = recall_score(y_true, y_pred, average='macro')

        f1_score_weighted = f1_score(y_true, y_pred, average = 'weighted')
        precision_score_weighted = precision_score(y_true, y_pred, average = 'weighted')
        recall_score_weighted = recall_score(y_true, y_pred, average = 'weighted')
        print("accuracy score: "+str(accuracyScore))
        print("roc auc score: "+str(rocaucscore))
        print("balanced_accuracy_score: " + str(balancedAccuracyScore))
        print("\n")
        print("Binary scores:\n")
        print("f1_score: "+str(f1score))
        print("precision score: "+str(precisionScore))
        print("recall score: "+str(recallScore))
        print("\n")
        print("Macro scores:\n")
        print("recall_score_macro: "+str(recall_score_macro))
        print("f1_score_macro: "+str(f1_score_macro))
        print("precision_score_macro: "+str(precision_score_macro))
        print("\n")
        print("Weighted Scores:\n")
        print("recall_score_weighted: "+str(recall_score_weighted))
        print("f1_score_weighted: "+str(f1_score_weighted))
        print("precision_score_weighted: "+str(precision_score_weighted))