import pandas as pd
import numpy as np
import math as math
import warnings

warnings.filterwarnings("ignore")


train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

species_group = train.groupby(['4'])

class1 = species_group.get_group('Iris-setosa')
class2 = species_group.get_group('Iris-versicolor')
class3 = species_group.get_group('Iris-virginica')


class1.drop(['4'], inplace=True, axis=1)
class2.drop(['4'], inplace=True, axis=1)
class3.drop(['4'], inplace=True, axis=1)


def find_mean(X):

    return np.array(np.mean(X))


classes = [class1, class2, class3]
def classifier(X):
    all_prob = []
    i=0
    for class_ in classes:
        cov = np.cov(class_.values.T)
        mu = find_mean(class_)
        size = len(X)
        det = np.linalg.det(cov)
        norm_const = 1.0/(math.pow((2*np.pi), float(size)/2) * math.pow(det, 1.0/2))
        x_mu = np.matrix(X - mu)
        inv = np.linalg.inv(cov)
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        all_prob.append(result*norm_const*prior[i])
        i=i+1
    if (all_prob.index(max(all_prob))) == 0:
        return 'Iris-setosa'
    if (all_prob.index(max(all_prob))) == 1:
        return 'Iris-versicolor'
    if (all_prob.index(max(all_prob))) == 2:
        return 'Iris-virginica'
    


def predict_target(feature_vectors):
    target_class_predicted = []
    for i in range(len(feature_vectors)):
        target_class_predicted.append(classifier(feature_vectors[i]))
    return target_class_predicted


correct = 0


for flag in (predict_target(test.T[:4].T.values) == test['4'].values):
    if flag:
        correct += 1

print("Accuracy is Bayes classifier is", (correct/test.shape[0])*100)
