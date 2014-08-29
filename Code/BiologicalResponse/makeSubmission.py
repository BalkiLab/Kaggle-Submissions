import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier, GradientBoostingClassifier

def evaluateModel(y,X,ncv,model):
    Xcv = KFold(len(X),ncv,indices = False)
    results = []
    for trainIdx,testIdx in Xcv:
        model.fit(X[trainIdx],y[trainIdx])
        results.append(log_loss(y[testIdx],model.predict_proba(X[testIdx])))   
    return results

def readData():
    trainData = pd.read_csv('Data/BiologicalResponse/train.csv')
    y = trainData.Activity.values
    X = trainData.drop('Activity',axis = 1).values
    Xtest = pd.read_csv('Data/BiologicalResponse/test.csv').values
    return y,X,Xtest

def makeSubmission(model,Xtest):
    predicted_probs = model.predict_proba(Xtest)[:,1]
    predicted_probs = (predicted_probs - predicted_probs.min()) / (predicted_probs.max() - predicted_probs.min())
    predicted_probs = pd.Series(predicted_probs)
    predicted_probs.index += 1
    predicted_probs.index.name = 'MoleculeId';
    predicted_probs.name = 'PredictedProbability'
    return predicted_probs

def testRandomForestModel(y,X,Xtest):
    numTrees = 100
    rfModel = RandomForestClassifier(numTrees,n_jobs = -1,min_split=2)
    results = evaluateModel(y, X, 5, rfModel)
    print "Results: " + str( np.array(results).mean() )
    rfModel.fit(X,y);
    predicted_probs=makeSubmission(rfModel, Xtest)
    predicted_probs.to_csv('Submissions/BiologicalResponse/rf'+ str(numTrees) +'submission.csv',header = True,index = True,float_format = "%f");


def getFolds(y, X,nFolds):
    idx = np.random.permutation(y.size)
    Xs = X[idx]
    ys = y[idx]
    skf = list(StratifiedKFold(y, nFolds))
    return Xs,ys,skf
    

def getBlendedModelFeatures(y, X, Xtest,nFolds):
    Xs,ys,skf = getFolds(y, X,nFolds)
    models = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]
    Xb = np.zeros((Xs.shape[0], len(models)))
    Xbtest = np.zeros((Xtest.shape[0], len(models)))
    for j,model in enumerate(models):
        Xbtest_j = np.zeros((Xtest.shape[0],len(skf)))
        for i,(trainIdx,testIdx) in enumerate(skf):
            model.fit(Xs[trainIdx],ys[trainIdx])
            Xb[testIdx,j] = model.predict_proba(Xs[testIdx])[:,1]
            Xbtest_j[:,i] = model.predict_proba(Xtest)[:,1]
        Xbtest[:,j] = Xbtest_j.mean(1)
    return Xb,ys,Xbtest
    
    

def testLogisticRegression(y, X, Xtest):
    lrModel = LogisticRegression(C=0.3,penalty='l1')
    results = evaluateModel(y, X, 5, lrModel)
    print "Results: " + str( np.array(results).mean() )
    lrModel.fit(X,y);
    predicted_probs=makeSubmission(lrModel, Xtest)
    predicted_probs.to_csv('Submissions/BiologicalResponse/lrsubmission.csv',header = True,index = True,float_format = "%f");


def testBlendedModel(y, X, Xtest):
    Xb,ys,Xbtest = getBlendedModelFeatures(y, X, Xtest, 10)
    lrModel = LogisticRegression()
    results = evaluateModel(ys, Xb, 5, lrModel)
    print "Results: " + str( np.array(results).mean() )
    lrModel.fit(Xb,ys);
    predicted_probs=makeSubmission(lrModel, Xbtest)
    predicted_probs.to_csv('Submissions/BiologicalResponse/blendedsubmission.csv',header = True,index = True,float_format = "%f");
    
def main():
    y,X,Xtest = readData()
    testRandomForestModel(y, X, Xtest)
    testLogisticRegression(y, X, Xtest)
    testBlendedModel(y, X, Xtest)
    
if __name__ == "__main__":
    np.random.seed(0) # seed to shuffle the train set
    main()


