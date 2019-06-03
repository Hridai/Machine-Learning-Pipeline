import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#For Classification "Social_Network_Ads.csv" for Regression "Salary_Data.csv" for polynomial regression "Position_Salaries.csv"
classification_model_names = [ "Logistic", "KNN", "SVM", "Bayes", "DecisionTree", "RandomForest" ] # Maintain this list

preprocess_dict = {
        "TrainSplit": 0.25,
        "ModelType": "Classification",
        "ModelName": "Poly"
        }

flags_dict = {
        "do_feature_scaling" : True,
        "show_confusionmatrix" : True,
        "show_graph" : True,
        "do_all_classification_models" : True
        }

#### Model Hyperparameters
## Common
rand_state = 42
## Poly regression
poly_degree = 4
## K-Nearest-Neighbors
knn_neighbors = 5
knn_metric = "minkowski"
knn_minkowski_power = 2
## Support Vector Machine
svc_kernel = "rbf"
## Trees
tree_criterion = "entropy"
tree_n = 10
                
def runclassification( model_name_in ):
    if( model_name_in == 'Logistic' ):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression( random_state = rand_state )
        classifier.fit( X_train, Y_train )
        Y_pred = classifier.predict( X_test )
        
    elif( model_name_in == "KNN" ):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier( n_neighbors = knn_neighbors, metric = knn_metric, p = knn_minkowski_power )
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
    
    elif( model_name_in == "SVM" ):
        from sklearn.svm import SVC
        classifier = SVC( kernel = svc_kernel, random_state = rand_state )
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        
    elif( model_name_in == "Bayes" ):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        
    elif( model_name_in == "DecisionTree" ):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier( criterion = tree_criterion, random_state = rand_state )
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
    
    elif( model_name_in == "RandomForest" ):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier( n_estimators = tree_n, criterion = tree_criterion, random_state = rand_state )
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)

    if( flags_dict["show_confusionmatrix"] ):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix( Y_test, Y_pred )
        print( model_name_in, 'confusion matrix \n', cm)
    
    if( flags_dict["show_graph"] ):
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_test, Y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title( model_name_in )
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()

def run_regression( model_name_in ):
    if( preprocess_dict["ModelName"] == "Poly" ):
        from sklearn.linear_model import LinearRegression
        lin_reg = LinearRegression()
        lin_reg.fit( X, Y )
        from sklearn.preprocessing import PolynomialFeatures
        poly_reg = PolynomialFeatures( degree = poly_degree )
        X_poly = poly_reg.fit_transform(X)
        poly_reg.fit( X_poly, Y )
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit( X_poly, Y )
    
    if( flags_dict["show_graph"] ):
        plt.scatter(X_test, Y_test, color = 'red')
        X_grid = np.arange(min(X), max(X), 0.1)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, Y, color = 'red')
        plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
        plt.title('Salary vs Experience (Test set)')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.show()

##### Main #####
filepath="C:\\Users\Hridai\Desktop\Machine Learning A-Z Udemy\\"
filename="Social_Network_Ads.csv"
filepath = filepath + filename
dataset = pd.read_csv( filepath )
X = dataset.iloc[:, [2,3] ].values
Y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = preprocess_dict["TrainSplit"], random_state = rand_state )
if( flags_dict["do_feature_scaling"] ):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test) 
if( preprocess_dict["ModelType"] == 'Classification' ):
    if( flags_dict["do_all_classification_models"] ):
        for name in classification_model_names:
            temp_model_name = name
            runclassification( temp_model_name )
    else:
        runclassification( preprocess_dict["ModelName"] )
if( preprocess_dict["ModelName"] == "Regression" ):        
    run_regression( flags_dict["ModelName"] )