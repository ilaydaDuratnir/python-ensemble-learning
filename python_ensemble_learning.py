#eng:import library
#tr:kütüphanelerin yüklenmesi
import numpy as np # eng: mathematical functions, tr:matematiksel işlemler 
import pandas as pd #eng:data analysis and manipulation tool, tr:dataframeler ile veri analizi işlemleri
import seaborn as sns# eng:data visualization library, tr:görselleştirme 

import matplotlib.pyplot as plt# eng:data visualization library, tr:görselleştirme
from matplotlib.colors import ListedColormap# eng:data visualization library, tr:görselleştirme

from sklearn.model_selection import train_test_split# eng:separation of the data set as training and testing , tr:veri setinin eğitim ve test olarak ayrılması
from sklearn.preprocessing import RobustScaler #eng:Scale features using statistics that are robust to outliers, tr:Aykırı değerlerden etkilenmeden öznitelik ölçekleme yapılması
from sklearn.datasets import make_moons, make_circles, make_classification #eng:create datasets 2 binary 1 multiclass, tr: verisetleri oluşturma 2 ikili 1 çoklu sınıf
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier #eng:ensemble learnig , tr:topluluk öğrenmesi

# eng:warning library, tr:uyarı kütüphanesi
import warnings
warnings.filterwarnings('ignore') 

#eng:create datasets
#tr:veri setlerini oluşturma
random_state = 42



n_samples = 1000 #eng:number of samples per cluster, tr:küme başına örnek sayısı
n_features = 10 #eng:number of features for each sample, tr:her bir örnek için özellik sayısı
n_classes = 2 #eng:number of classes (or labels), tr:sınıfların (veya etiketlerin) sayısı

#eng:Larger values introduce noise in the labels and make the classification task harder
#tr:Daha büyük değerler etiketlerde parazite neden olur ve sınıflandırmayı zorlaştırır
noise_moon = 0.3
noise_circle = 0.3
noise_class = 0.3 

X,y = make_classification(n_samples = n_samples,
                    n_features = n_features,
                    n_classes = n_classes,
                    n_repeated = 0,#eng:number of duplicated features, tr:tekrar eden özellik sayısı
                    n_redundant = 0,#eng:number of redundant features, tr:anlamsız örnek sayısı
                    n_informative = n_features-1,#eng:number of informative features, tr:bilgilendirici özelliklerin sayısı
                    random_state = random_state,
                    n_clusters_per_class = 1,#eng:number of clusters per class, tr:sınıf başına küme sayısı
                    flip_y = noise_class)


data = pd.DataFrame(X)
data["target"] = y
plt.figure()
sns.scatterplot(x = data.iloc[:,0], y =  data.iloc[:,1], hue = "target", data = data ) #eng:visualization, tr:görselleştirelim

data_classification = (X,y)

moon = make_moons(n_samples = n_samples, noise = noise_moon, random_state = random_state)

#data = pd.DataFrame(moon[0])
#data["target"] = moon[1]
#plt.figure()
#sns.scatterplot(x = data.iloc[:,0], y =  data.iloc[:,1], hue = "target", data = data ) #eng:visualization, tr:görselleştirelim

circle = make_circles(n_samples = n_samples, factor = 0.1,  noise = noise_circle, random_state = random_state)

#data = pd.DataFrame(circle[0])
#data["target"] = circle[1]
#plt.figure()
#sns.scatterplot(x = data.iloc[:,0], y =  data.iloc[:,1], hue = "target", data = data ) #eng:visualization, tr:görselleştirelim

datasets = [moon, circle]
 
# Basic Classifiers : KNN, SVM, DT
n_estimators = 10 #♀eng:number of trees in the forest, tr:ağaç sayısı

svc = SVC()
knn = KNeighborsClassifier(n_neighbors = 15)
dt = DecisionTreeClassifier(random_state = random_state, max_depth = 2)

rf = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, max_depth = 2)
ada = AdaBoostClassifier(base_estimator = dt, n_estimators = n_estimators, random_state = random_state)
v1 = VotingClassifier(estimators = [('svc',svc),('knn',knn),('dt',dt),('rf',rf),('ada',ada)])

names = ["SVC", "KNN", "Decision Tree", "Random Forest", "AdaBoost", "V1"]
classifiers = [svc, knn, dt, rf, ada, v1]

h=0.2 #eng:resolution, tr:çözünürlük
i = 1
figure = plt.figure(figsize=(18, 6))
#eng:Training of algorithms and visualization of results
#tr: Algoritmaların eğitimi ve sonuçların görselleştirilmesi
for ds_cnt, ds in enumerate(datasets): #datasets --> circle and moon , ds_cnt --> index
    #eng: preprocess dataset, split into training and test part
    #tr: veri setinin ön işlemesi, eğitim ve teste bölünür
    X, y = ds
    X = RobustScaler().fit_transform(X)#eng:Scale features using statistics that are robust to outliers, tr:Aykırı değerlerden etkilenmeden öznitelik ölçekleme yapılması
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=random_state)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    #meshgrid --> eng:used to create a rectangular grid out of two given one-dimensional arrays representing the Cartesian indexing or Matrix indexing
    #tr: Kartezyen indekslemeyi veya Matris indekslemeyi temsil eden iki tek boyutlu diziden dikdörtgen bir ızgara oluşturmak için kullanılır
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    #eng:colors for visualization ,tr:görselleştirme renkleri
    cm = plt.cm.RdBu #eng:visualization tr:
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
                                
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i) # len(datasets) : row(satır), len(classifiers) +1 : column(sütun)
    
    if ds_cnt == 0:
        ax.set_title("Input data")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k') #eng:visualization, tr:görselleştirelim
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,marker = '^', edgecolors='k') #eng:visualization, tr:görselleştirelim
    
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    print("Dataset # {}".format(ds_cnt))
          
    # classifiers : KNN , SVC , DT
    for name, clf in zip(names, classifiers):
        
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        
        clf.fit(X_train, y_train) 
        
        score = clf.score(X_test, y_test)
        
        print("{}: test set score: {} ".format(name, score))
        
        score_train = clf.score(X_train, y_train)  
        
        print("{}: train set score: {} ".format(name, score_train))
        print()
        
        #eng:The hasattr() method returns true if an object has the given named attribute and false if it does not.
        #tr: Hasattr () yöntemi, bir nesne belirtilen adlandırılmış özelliğe sahipse true, yoksa false döndürür.
        if hasattr(clf, "decision_function"):
            #ravel --> eng: Return a contiguous flattened array, tr:Bitişik düzleştirilmiş bir dizi döndürür
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])#np.c --> eng:concatenation , tr: birleştirme
        else:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # eng :Put the result into a color plot, tr:Sonucu bir renk grafiğine koyuyoruz
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        #eng:Plot the training points, tr:Eğitim noktalarını belirliyoruz
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        #eng:Plot the testing points, tr:Test noktalarını belirliyoruz
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,marker = '^',
                   edgecolors='white', alpha=0.6)

        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        score = score*100
        ax.text(xx.max() - .3, yy.min() + .3, ('%.1f' % score),
                size=15, horizontalalignment='right')
        i += 1
    print("-------------------------------------")

plt.tight_layout()
plt.show()

def make_classify(dc, clf, name):
    x, y = dc
    x = RobustScaler().fit_transform(x)#eng:Scale features using statistics that are robust to outliers, tr:Aykırı değerlerden etkilenmeden öznitelik ölçekleme yapılması
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.4, random_state=random_state)
    
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("{}: test set score: {} ".format(name, score))
        score_train = clf.score(X_train, y_train)  
        print("{}: train set score: {} ".format(name, score_train))
        print()

print("Dataset # 2")   
make_classify(data_classification, classifiers,names)  






















