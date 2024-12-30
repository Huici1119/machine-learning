import numpy as np
import sklearn
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn import  ensemble, preprocessing, metrics
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

#讀取資料 
df = pd.read_csv("D:\\HC\\碩士班課程\\機器學習原理\\Student Depression Dataset.csv")
path = "D:\\HC\\碩士班課程\\機器學習原理\\" #設定存檔路徑
