from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from  pgmpy.inference.EliminationOrder import WeightedMinFill
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.estimators import BicScore
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_score, recall_score, roc_auc_score, balanced_accuracy_score

class ModelEvaluation:
    
    
    def trainModel(training_data: pd.DataFrame):
        
        # structure Learning
        
        scoring_method = BicScore(data=training_data)
        est = HillClimbSearch(data=training_data, use_cache = True)
        estimated_model = est.estimate(
        scoring_method=scoring_method, max_iter=int(1e4))
        model = BayesianNetwork(estimated_model.edges())
        
        active_trail_nodes = model.active_trail_nodes('class')['class']
        active_trail_nodes_list = list(active_trail_nodes)
        original_nodes = list(model.nodes())
        for node in original_nodes:
            if node not in active_trail_nodes_list:
                model.remove_node(node)
        
        # Parameter Estimation
        

        estimator = MaximumLikelihoodEstimator(model,training_data)
        parameters = estimator.get_parameters(n_jobs=6)

        for i in range(len(parameters)):
            model.add_cpds(parameters[i])
            #print(parameters[i])
        
        return model
    
    
    def performInference(model: BayesianNetwork, testing_evidence: list, testing_targets: pd.Series):
        
        for i in range(len(testing_evidence)):
            for attribute in list(testing_evidence[i].keys()):
                if attribute not in model.nodes():
                    del testing_evidence[i][attribute]
            

        inference = VariableElimination(model)

        y_pred = []
        y_true = []
        for i in range(len(testing_evidence)):
            bunkrupt_or_not = inference.map_query(['class'], evidence=testing_evidence[i],show_progress=False)
            y_pred.append(bunkrupt_or_not['class'])
            y_true.append(testing_targets.iloc[i])
        
        return y_pred, y_true
    
    def evaluatePrint(y_pred:list, y_true: list):
        
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
        
    
        
        
    # The following code was adapted from https://medium.com/@avijit.bhattacharjee1996/implementing-k-fold-cross-validation-from-scratch-in-python-ae413b41c80d    
        
        
    def kfold_indices(data: pd.DataFrame, k: int):
        fold_size = len(data) // k
        indices = np.arange(len(data))
        folds = []
        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            folds.append((train_indices, test_indices))
        return folds
        
    def perfrom_KfoldCrossValidation(folds: list, data: pd.DataFrame):
        from sklearn.metrics import accuracy_score
        from sklearn.linear_model import LogisticRegression  # Replace with your model of choice

        # Initialize your machine learning model (e.g., Logistic Regression)
        model = LogisticRegression()

        # Initialize a list to store the evaluation scores
        accuracyScore_list = []
        balancedAccuracyScore_list = []
        f1score_list = []
        precisionScore_list  = []
        recallScore_list  = []
        
        f1scoresWeighted_list  = []
        precisionScoresWeighted_list  = []
        recallScoresWeighted_list  = []
        
        # Iterate through each fold
        for train_indices, test_indices in folds:
            training_data = data.iloc[train_indices]
            testing_data = data.iloc[test_indices]
            
            
            testing_targets = testing_data["class"]
            del testing_data["class"]

            testing_evidence_list = []
            for i in range(len(testing_data)):
                testing_evidence_dict = {}
                for z in range(len(testing_data.columns.values.tolist())):
                    testing_evidence_dict[testing_data.columns.values.tolist()[z]] = testing_data[testing_data.columns.values.tolist()[z]].iloc[i]
                testing_evidence_list.append(testing_evidence_dict)

            
            # Train the model on the training data
            try:
                model = ModelEvaluation.trainModel(training_data)
            except:
                print("An error occurred during either structure learning or parameter estimation.")
                continue
            # Make predictions on the test data
            try:
                y_pred, y_true = ModelEvaluation.performInference(model = model, testing_evidence = testing_evidence_list, testing_targets=testing_targets)
            except:
                print("An Error occurred during inference.")
                continue
            
            # Calculate the accuracy score for this fold
            accuracyScore = accuracy_score(y_true, y_pred)
            balancedAccuracyScore = balanced_accuracy_score(y_true, y_pred,adjusted=True)
            f1score = f1_score(y_true, y_pred)
            precisionScore = precision_score(y_true, y_pred)
            recallScore = recall_score(y_true, y_pred)
            
            f1_score_weighted = f1_score(y_true, y_pred, average = 'weighted')
            precision_score_weighted = precision_score(y_true, y_pred, average = 'weighted')
            recall_score_weighted = recall_score(y_true, y_pred, average = 'weighted')

            # Append the fold score to the list of scores
            accuracyScore_list.append(accuracyScore)
            balancedAccuracyScore_list.append(balancedAccuracyScore)
            f1score_list.append(f1score)
            precisionScore_list.append(precisionScore)
            recallScore_list.append(recallScore)
            
            f1scoresWeighted_list.append(f1_score_weighted)
            precisionScoresWeighted_list.append(precision_score_weighted)
            recallScoresWeighted_list.append(recall_score_weighted)

        # Calculate the mean accuracy across all folds
        mean_accuracy = np.mean(accuracyScore_list )
        mean_balanced_accuracy = np.mean(balancedAccuracyScore_list)
        mean_f1 = np.mean(f1score_list )
        mean_precision = np.mean(precisionScore_list )
        mean_reall = np.mean(recallScore_list )
        
        mean_f1_weighted = np.mean(f1scoresWeighted_list )
        mean_precision_weighted = np.mean(precisionScoresWeighted_list )
        mean_reall_weighted = np.mean(recallScoresWeighted_list)
        print("Mean Accuracy:", mean_accuracy.round(4))
        print("Mean Balancedd Accuracy:", mean_balanced_accuracy.round(4))
        print("Mean F1:", mean_f1.round(4))
        print("Mean Precision:", mean_precision.round(4))
        print("Mean Recall:", mean_reall.round(4))
        
        print("Mean Weighted F1:", mean_f1_weighted.round(4))
        print("Mean Weighted Precision:", mean_precision_weighted.round(4))
        print("Mean Weighted Recall:", mean_reall_weighted.round(4))