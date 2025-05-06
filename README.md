# Nover-Project
Data mining Uts

import pandas as pd / untuk menyatakan fungsi Pandas sebagai PD
from google.colab import files
uploaded = files.upload() = Menggunakan ini untuk mengupload dataset yang akan digunakan ke google collab

data1 = pd.read_csv("MQTT DoS Publish Flood.csv") = Untuk menyatakan variabel data1 menggunakan dataset "MQTT DoS Publish Flood.csv"

data2 = pd.read_csv("DDoS ICMP Flood.csv") = untuk menyatakan variabel data 2 menggunakan dataset "DDoS ICMP Flood.csv

x = hasilgabung.iloc[:,7: 76] 
y = hasilgabung.iloc[:,83: 84]  

=
 untuk mengambil subset kolom dari sebuah DataFrame bernama hasilgabung menggunakan indexing dengan iloc, yang artinya index berbasis angka (bukan label/kolom nama).

 from sklearn.model_selection import train_test_split = pemanggilan fungsi sklearn untuk mentraining dataset

 x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42) = merupakan cara untuk membagi dataset menjadi data latih dan data uji menggunakan fungsi train_test_split dari pustaka sklearn.model_selection

 from sklearn import tree
from sklearn.tree import DecisionTreeClassifier = berfungsi untuk mengimpor modul dan kelas dari pustaka Scikit-learn yang digunakan untuk membuat model Decision Tree (Pohon Keputusan)

alya = DecisionTreeClassifier(criterion='entropy', splitter = 'random')
alya.fit(x_train,y_train) = untuk membuat dan melatih model Decision Tree Classifier dengan pengaturan khusus.

y_pred = alya.predict(x_test) = proses melakukan prediksi menggunakan model pohon keputusan (alya) yang telah dilatih sebelumnya

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) = digunakan untuk mengukur akurasi dari model klasifikasi, seberapa banyak prediksi model yang benar dibandingkan dengan label yang sebenarnya.

import matplotlib.pyplot as plt
import numpy as np = fungsi pengimporan dua pustaka penting dalam Python yang biasa digunakan untuk visualisasi data dan komputasi numerik.

fig = plt.figure(figsize = (10, 7))
tree.plot_tree(alya, feature_names = x.columns.values, class_names = np.array([ 'Benign Traffic','DDos ICMP Flood','DDoS UDP Flood']), filled = True)
plt.show() = digunakan untuk visualisasi pohon keputusan yang telah dilatih, yaitu model alya.

import seaborn as lol
from sklearn import metrics
label = np.array([ 'MQTT DoS Publish Flood', 'DDoS ICMP Flood']) = adalah cara untuk mengimpor pustaka yang digunakan untuk visualisasi dan evaluasi model.

import matplotlib.pyplot as plt
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
lol.heatmap(conf_matrix, annot=True, xticklabels=label, yticklabels=label)
plt.xlabel('Prediksi')
plt.ylabel('Fakta')
plt.show() = untuk menampilkan visualisasi confusion matrix yang menunjukkan seberapa baik model dalam memprediksi kelas dengan menggunakan heatmap.
