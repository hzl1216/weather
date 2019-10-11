import random
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def k_fold(total,n):
    all_index = [ i for  i in range(total)]
    k_folds = [ ([],[]) for i in  range(n) ]
    for i in range(total):
        k = random.randint(0,n-1)
        k_folds[k][1].append(i)
        for j in range(n):
            if j !=k:
                k_folds[j][0].append(i)
    return k_folds
if __name__ =='__main__':
    
    print((len(k_fold(6391,5)[1][0])))
        
        
    
