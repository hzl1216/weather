import  pandas as pd
import os
from collections import Counter

def most(nums):
    return  Counter(nums).most_common(1)[0][0]
def most2(nums):
    return  Counter(nums).most_common(1)[0]
def get_model_result_1(model_names,submit_type):
    files= os.listdir('result')
    times = 5
    submits=[]
    types = []
    for model_name in model_names:
        model_types=[]
        model_submits = []
        for i in range(times):
            s =[]
#            for file in files:
#                if 'submit_%s_%d'%(model_name,i) in file:
#                    s.append(int(file[len('submit_%s_%d_'%(model_name,i)):].split('.')[0]))
            submit = pd.read_csv('result/submit_%s_%d_%s.csv'%(model_name,i,submit_type))
            model_submits.append(submit['type'])
            file_names = submit['FileName']
            length = len(submit)
        for i in range(length):
            nums = [model_submits[j][i] for j in range(times)]
            model_types.append(most(nums))
        submits.append(model_types)
    for i in range(length):
#        weight = [0 for _ in range(10)]
#        for j in range(len(model_names)):
#            weight[submits[j][i]]+= models_weight[j]
#        types.append(weight.index(max(weight)))
        nums = [submits[j][i] for j in range(len(model_names))]
        types.append(most(nums))

    dataframe = pd.DataFrame({'FileName': file_names, 'type': types})
    dataframe.to_csv('result/submit.csv', index=False, sep=',')

def get_model_result_2(model_names,submit_type):
    times=5
    files= os.listdir('result')
    submits=[]
    types = []
    for model_name in model_names:
        for i in range(times):
            s =[]
            submit = pd.read_csv('result/submit_%s_%d_%s.csv'%(model_name,i,submit_type))
            submits.append(submit['type'])
            file_names = submit['FileName']
            length = len(submit)
    for i in range(length):
        nums = [submits[j][i] for j in range(times*len(model_names))]
        types.append(most(nums))
    dataframe = pd.DataFrame({'FileName': file_names, 'type': types})
    dataframe.to_csv('result/submit.csv', index=False, sep=',')


def get_model_result_3(model_names,submit_type):
    files= os.listdir('result')
    times = 5
    submits=[]
    types = []
    for model_name in model_names:
        submit = pd.read_csv('result/%s_submit.csv'%(model_name))
        submits.append(submit['type'])
        file_names = submit['FileName']
        length = len(submit)
    for i in range(length):
        weight = [0 for _ in range(10)]
        for j in range(len(model_names)):
            print(submits[j][i])
            weight[submits[j][i]]+= models_weight[j]
        types.append(weight.index(max(weight)))

    dataframe = pd.DataFrame({'FileName': file_names, 'type': types})
    dataframe.to_csv('result/submit.csv', index=False, sep=',')
def generator_label(model_names,submit_type):
    files= os.listdir('result')
    submits=[]
    types = []
    filenames = []
    for model_name in model_names:
        for i in range(5):
            s =[]
            submit = pd.read_csv('result/submit_%s_%d_%s.csv'%(model_name,i,submit_type))
            submits.append(submit['type'])
            file_names = submit['FileName']
            length = len(submit)
    for i in range(length):
        nums = [submits[j][i] for j in range(5*len(model_names))]
        value , times = most2(nums)
        if times==5*len(model_names):
            types.append(value)
            filenames.append(file_names[i])
    dataframe = pd.DataFrame({'FileName': filenames, 'type': types})
    dataframe.to_csv('other_labels.csv', index=False, sep=',')
def resort(model_names,submit_type='best'):
    times = 5
    file_names = pd.read_csv('result/submit.csv' )['FileName']
    for model_name in model_names:
        for i in range(times):
            table = 'result/submit_%s_%d_%s.csv' % (model_name, i, submit_type)
            submit = pd.read_csv(table)
            types = []
            for file_name in file_names:
                type.append(submit['type'][submit['FileName'].index(file_name)])

            length = len(submit)

            dataframe = pd.DataFrame({'FileName': file_names, 'type': types})
            dataframe.to_csv(table, index=False, sep=',')

if __name__ =='__main__':
#models_names=['se_resnet101','efficientnet-b4','se_resnext50_32x4d','efficientnet-b5','se_resnet50']
    models_names = ['senet154']

    models_weight = [4,3,2]
    get_model_result_3(models_names,'best')
#    generator_label(models_names,'best')
