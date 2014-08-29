import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    trainData = pd.read_csv('Data/BiologicalResponse/train.csv');
    
    y = trainData.Activity.values
    X = trainData.drop('Activity',axis=1).values
    Xnew = pd.read_csv('Data/BiologicalResponse/test.csv').values
    randomForestModel = RandomForestClassifier(n_estimators=100,n_jobs=-1)
    randomForestModel.fit(X,y);
    predicted_probs = [x[1] for x in randomForestModel.predict_proba(Xnew)]
    predicted_probs = pd.Series(predicted_probs)
    predicted_probs.index +=1
    predicted_probs.index.name = 'MoleculeId';
    predicted_probs.name = 'PredictedProbability'
    predicted_probs.to_csv('Submissions/BiologicalResponse/rf100submission.csv',header=True,index=True,float_format="%f");

if __name__=="__main__":
    main()
    
    