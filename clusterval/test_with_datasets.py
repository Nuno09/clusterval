from sklearn.datasets import load_iris, make_blobs, load_breast_cancer, load_wine
from evaluate import Clusterval
from datasets.datasets import load_khan_train, load_khan_test, load_vote_repub, load_animals
import re

synthetic_500_k_4, _ = make_blobs(n_samples=500, centers=4, n_features=5, random_state=0)

synthetic_dim2_9 = []
pattern = re.compile(r'^\s+')
with open('/home/nuno/Documentos/Datasets/data_dim_k=9_txt/dim2.txt', 'r') as dim2:
    for line in dim2:
        re_new = re.sub(pattern, '', line)
        new = ''
        for i, el in enumerate(re_new[:-1]):
            if (el != ' ') and (re_new[i+1] == ' '):
                new = new + el + ','
            elif el != ' ':
                new += el

        new = new.split(',')
        new = [float(item) for item in new]
        synthetic_dim2_9.append(new)

synthetic_dim3_9 = []
with open('/home/nuno/Documentos/Datasets/data_dim_k=9_txt/dim3.txt') as s1:
    for line in s1:
        re_new = re.sub(pattern, '', line)
        new = ''
        for i, el in enumerate(re_new[:-1]):
            if (el != ' ') and (re_new[i + 1] == ' '):
                new = new + el + ','
            elif el != ' ':
                new += el

        new = new.split(',')
        new = [float(item) for item in new]
        synthetic_dim3_9.append(new)

datasets = {'iris': load_iris()['data'], 'wine': load_wine()['data'], 'cancer': load_breast_cancer()['data'],
            'synthetic': synthetic_500_k_4, 'train_khan': load_khan_train()['data'], 'test_khan': load_khan_test()['data'],
            'vote.repub': load_vote_repub()['data'], 'animals': load_animals()['data'], 's_dim2_k_9': synthetic_dim2_9,
            's_dim3_k_9': synthetic_dim3_9}

linkage = ['ward', 'single', 'centroid', 'average', 'complete']



c = Clusterval(min_k=6, max_k=12)

for name, dataset in datasets.items():
    if name == 's_dim3_k_9':
        print('Testing for {0}\n'.format(name))
        for l in linkage:
            print('{0} with linkage {1}\n'.format(name, l))
            c.link = l
            eval = c.evaluate(dataset)
            print(eval.long_info)
        print('\n\n')

