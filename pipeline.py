# Machine Learning Pipeline. Hridai Trivedy. May 2019. hridaitrivedy@gmail.com

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#For Classification "Social_Network_Ads.csv" for Regression "Salary_Data.csv" for polynomial regression "Position_Salaries.csv"
classification_model_names = [ "Logistic", "KNN", "SVM", "Bayes", "DecisionTree", "RandomForest" ] # Maintain this list

preprocess_dict = {
        "TrainSplit": 0.25,
        "ModelType": "Classification",
        "ModelName": "KNN",
        }

flags_dict = {
        "show_confusionmatrix" : True,
        "show_graph" : True,
        "do_all_classification_models" : True,
        "calc_MSE" : True,
        }

rand_state = 42 # Rand seed number. Write "None" for true randomness


class MSE_pair:
    def __init__(self, in_model_name, in_mse_value):
        self.MSE_model_name = in_model_name
        self.MSE_value = in_mse_value

    def showMSE(self):
        print( "MSE for ", self.MSE_model_name, " is:[" , self.MSE_value, "]")


def run_classification( model_name_in, mse_vec ):
    if( model_name_in == 'Logistic' ):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression( random_state = rand_state )
        classifier.fit( X_train, Y_train )
        Y_pred = classifier.predict( X_test )
        
    elif( model_name_in == "KNN" ):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier( n_neighbors = 5, metric = "minkowski", p = 2 )
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
    
    elif( model_name_in == "SVM" ):
        from sklearn.svm import SVC
        classifier = SVC( kernel = 'rbf', random_state = rand_state )
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        
    elif( model_name_in == "Bayes" ):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        
    elif( model_name_in == "DecisionTree" ):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier( criterion = 'entropy', random_state = rand_state )
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
    
    elif( model_name_in == "RandomForest" ):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier( n_estimators = 10, criterion = 'entropy', random_state = rand_state )
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)

    if( flags_dict["show_confusionmatrix"] ):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix( Y_test, Y_pred )
        print( model_name_in, 'confusion matrix \n', cm)
    
    if( flags_dict["calc_MSE"] ):
        from sklearn.metrics import mean_squared_error
        final_mse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        if( flags_dict['do_all_classification_models']):
            msepair = MSE_pair( model_name_in, final_mse )
            mse_vec.append( msepair )
        else:
            print( "MSE for ", model_name_in, " is:[" , final_mse, "]")
        
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
    if( preprocess_dict["ModelName"] == "Linear" ): # Works
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit( X_train, Y_train )
        Y_pred = regressor.predict( X_test )
    
    elif( preprocess_dict["ModelName"] == "Poly" ): # Broken
        from sklearn.preprocessing import PolynomialFeatures
        regressor = PolynomialFeatures( degree = 2 )
        X_poly = regressor.fit_transform( X_train )
        regressor.fit( X_train, Y_train )
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit( X_poly, Y_train )
        Y_pred = lin_reg_2.predict( X_train )
    
    elif( preprocess_dict['ModelName'] == 'SVM' ): # Broken
        from sklearn.svm import SVR
        regressor = SVR( kernel = 'rbf' )
        regressor.fit( X_train, Y_train )
        # Predicting a new result
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        sc_Y = StandardScaler()
        Y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
        Y_pred = sc_Y.inverse_transform( Y_pred )
        
    elif( preprocess_dict['ModelName'] == 'DecisionTree' ): # Broken
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor( random_state = rand_state )
        regressor.fit( X, Y )
        Y_pred = regressor.predict(6.5)
    
    elif( preprocess_dict['ModelName'] == 'RandomForest' ): # Untested
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor( n_estimators = 20, random_state = rand_state )
        regressor.fit(X, Y)
        Y_pred = regressor.predict(6.5)
    
    if( flags_dict["show_graph"] and ( preprocess_dict["ModelName"] == "Poly" or preprocess_dict["ModelName"] == "Linear" ) ):
        plt.scatter(X_test, Y_test, color = 'red')
        plt.plot( X_train, regressor.predict(X_train), color = 'blue' )
        plt.plot( X_test, regressor.predict( X_test ), color = 'orange' )
        plt.title('Salary vs Experience (Test set)')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.show()
    
    elif( flags_dict["show_graph"] and ( ( preprocess_dict["ModelName"] == "SVM" ) or preprocess_dict['ModelName'] == 'DecisionTree' or 
         preprocess_dict['ModelName'] == 'RandomForest' )):
        # Visualising the SVR results (for higher resolution and smoother curve)
        X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, Y, color = 'red')
        plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
        plt.title('Truth or Bluff (SVR)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        plt.show()


##### Main #####
## Import file into pandas dataset
filepath="C:\\Users\Hridai\Desktop\Machine Learning A-Z Udemy\\"
filename="Social_Network_Ads.csv"
filename="Salary_Data.csv"
filepath = filepath + filename
dataset = pd.read_csv( filepath )


## Analyse import
dataset.head()
dataset.info()
disp = dataset.describe()
dataset.hist(bins=50)
from pandas.plotting import scatter_matrix
attributes = ["Gender", "Age", "EstimatedSalary", "Purchased"]
scatter_matrix( dataset[attributes])
corr_matrix = dataset.corr()
corr_matrix["Purchased"].sort_values( ascending=False )


## Clean data
## Dealing with a feature which has blanks or NA values
dataset.dropna(subset=["FEATURENAME"])
dataset.drop("FEATURENAME", axis=1 )
median = dataset["EstimatedSalary"].median()
dataset["EstimatedSalary"].fillna( median, inplace=True ) # fills in the median values instead of NA
# there exists a "from skylearn.impute import SimpleImputer" that does somethign like the above. Pg 61


## Split indep/dependent variables
X = dataset.iloc[:, 2:-1 ].values # we want this to be a numpy object to carry out encoding
Y = dataset.iloc[:,4].values


####DISP
X = dataset.iloc[ :,0].values
Y = dataset.iloc[:, -1:].values

## Encode categorial features. Note the X value cannot be a dataframe it must be a numpy object
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
onehotencoder = OneHotEncoder( categorical_features= [0] )
X = onehotencoder.fit_transform( X ).toarray()


## Random or Stratified sampling:
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = rand_state)
X_train, X_Test = split.split( X, X["INDEX"] )
# non-stratified sampling (random)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = preprocess_dict["TrainSplit"], random_state = rand_state )


## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform( X_train )
X_test = sc.transform( X_test )


## Gridsearch. Optimize hyperparameters. Note the existence of a class RandomizedSearchCV
## which randomises features and tests permutations of their hyperparameters.
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
gridmodel = RandomForestClassifier()
param_grid = [
        {'n_estimators':[1,2,3], 'criterion':['gini','entropy']}
        ]
gridsearch = GridSearchCV(gridmodel, param_grid, cv=5, scoring='neg_mean_squared_error')
gridsearch.fit( X_train, Y_train )
gridsearch.best_estimator_
gs_results = gridsearch.cv_results_
gs_feature_importance = gridsearch.best_estimator_.feature_importances_ # gives the relative importance of each feature
''' you can use the above line to filter out the least important characteristics '''

if( preprocess_dict["ModelType"] == 'Classification' ):
    if( flags_dict["do_all_classification_models"] ):
        mse_vec = []
        for name in classification_model_names:
            temp_model_name = name
            run_classification( temp_model_name, mse_vec )
        minMSE = 999999.
        minMSEname = ''
        for i in range( 0, len(mse_vec), 1):
            temp_mse_pair = mse_vec[i]
            temp_mse_pair.showMSE()
            if( temp_mse_pair.MSE_value < minMSE ):
                minMSE = temp_mse_pair.MSE_value
                minMSEname = temp_mse_pair.MSE_model_name
        print( 'min values are found for model [', minMSEname, '] with value [', minMSE, ']')
    else:
        run_classification( preprocess_dict["ModelName"], mse_vec )

if( preprocess_dict["ModelName"] == "Regression" ):        
    run_regression( flags_dict["ModelName"] )