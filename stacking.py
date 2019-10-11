import pandas as pd
from collections import Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
def most(nums):
    return  Counter(nums).most_common(1)[0][0]
def get_model_result(model_names,submit_type='best'):
    times = 5
    for model_name in model_names:
        submits = []
        types = []
        for i in range(times):

            submit = pd.read_csv('result/submit_%s_%d_%s.csv'%(model_name,i,submit_type))
            submits.append(submit['type'])
            file_names = submit['FileName']
            length = len(submit)
        for i in range(length):
            nums = [submits[j][i] for j in range(times)]
            types.append(most(nums))
        dataframe = pd.DataFrame({'FileName': file_names, 'type': types})
        dataframe.to_csv('result/%s_submit.csv'%model_name, index=False, sep=',')
def stacking(model_names):
    test_inputs = None
    train_inputs = None
    train_label = pd.read_csv('Train_label.csv')['type']
    for model_name in model_names:
        submit = pd.read_csv('result/%s_submit.csv'%model_name)

        if test_inputs is None:
            test_inputs=submit['type']
        else:
            test_inputs=np.vstack([test_inputs, submit['type']])
        train =  pd.read_csv('result/test_label_%s.csv'%model_name)
        if train_inputs is None:
            train_inputs=train['type']
        else:
            train_inputs=np.vstack([train_inputs, train['type']])
        file_names = submit['FileName']
    test_inputs=test_inputs.T
    train_inputs=train_inputs.T
    clf1 = RandomForestClassifier(n_estimators=100)
    clf2 = DecisionTreeClassifier(max_depth=4)
    clf3 = ExtraTreesClassifier(n_estimators=10)
#    clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    clf = clf3
    scores = cross_val_score(clf, train_inputs, train_label)
    print(scores.mean())
    clf = clf.fit(train_inputs, train_label)
    types = clf.predict(test_inputs)
    dataframe = pd.DataFrame({'FileName': file_names, 'type': types})
    dataframe.to_csv('result/submit.csv', index=False, sep=',')

if __name__ =='__main__':

    model_names = ['senet154','se_resnet101','efficientnet-b5']
    get_model_result(model_names)
    stacking(model_names)
