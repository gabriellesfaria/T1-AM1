import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Leitura do dataset via pandas
iris = pd.read_csv('iris.data', header=None)
iris.rename(columns={0:'sepala_altura',
                     1:'sepala_largura',
                     2:'petala_altura',
                     3:'petala_largura',
                     4:'classe'}, inplace=True)

# Separação do dataset para inserção dos dados na árvore
iris_data = iris[['sepala_altura', 'sepala_largura', 'petala_altura', 'petala_largura']]
iris_label = iris['classe']

# Definição dos valores de treino e teste para a árvore
X_train, X_test, y_train, y_test = train_test_split(iris_data, 
                                                    iris_label, 
                                                    test_size=0.3, 
                                                    random_state=1)

# Mostra o tamanho dos dados de treino e teste
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Classificação de treino da árvore
clf = tree.DecisionTreeClassifier() 
clf = clf.fit(X_train, y_train)     
iris_pred = clf.predict(X_test)

# Cálculo da acurácia
acuracia = metrics.accuracy_score(y_test, iris_pred)

# Hora das plotagens e prints
print(acuracia) # 0.9555555555555556
tree.plot_tree(clf)  # Plotagem original

# Outra plotagem, com mais detalhes
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris_data.columns.unique(),  
                                class_names=iris_label.unique(),  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
