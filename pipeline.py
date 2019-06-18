# Machine Learning Pipeline. Hridai Trivedy. May 2019. hridaitrivedy@gmail.com

import preprocess_funcs as prep
import pandas as pd

#For Classification "Social_Network_Ads.csv" for Regression "Salary_Data.csv" for polynomial regression "Position_Salaries.csv"
classification_model_names = [ "Logistic", "KNN", "SVM", "Bayes", "DecisionTree", "RandomForest" ] # Maintain this list

preprocess_dict = {
        "TrainSplit": 0.25,
        "ModelType": "Classification",
        "ModelName": "KNN",
        'ClusterName' : 'KMeans',
        'RandState' : 1,
        }

flags_dict = {
        "show_confusionmatrix" : True,
        "show_graph" : True,
        "do_all_classification_models" : True,
        "calc_MSE" : True,
        'do_clustering' : True,
        }

##### Main #####
## Import file into pandas dataset
filepath="C:\\Users\Hridai\Desktop\Machine Learning\\"
filename="Social_Network_Ads.csv"
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
X = dataset.iloc[:, [2,3] ].values # we want this to be a numpy object to carry out encoding. Keep X a matrix, Y a vector.
Y = dataset.iloc[:,4].values


## Encode categorial features. Note the X value cannot be a dataframe it must be a numpy object
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder( categorical_features= [3] )
X = onehotencoder.fit_transform( X ).toarray()


## Random or Stratified sampling:
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = preprocess_dict["RandState"])
X_train, X_Test = split.split( X, X["INDEX"] )
# non-stratified sampling (random)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = preprocess_dict["TrainSplit"], random_state = preprocess_dict["RandState"] )


## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_train = sc_X.fit_transform( X_train )
X_test = sc_X.transform( X_test )


## Run Clustering if necessary
if( flags_dict['do_clustering'] ):
    prep.run_clustering( preprocess_dict, flags_dict, X, Y )


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
            prep.run_classification( preprocess_dict, flags_dict, X_train, X_test, Y_train, Y_test, temp_model_name, mse_vec )
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
        prep.run_classification( preprocess_dict, flags_dict, X_train, X_test, Y_train, Y_test, preprocess_dict["ModelName"], mse_vec )

if( preprocess_dict["ModelName"] == "Regression" ):        
    prep.run_regression( preprocess_dict, flags_dict, X, Y, flags_dict["ModelName"] )