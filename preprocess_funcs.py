# Machine Learning Pipeline. Hridai Trivedy. May 2019. hridaitrivedy@gmail.com
import matplotlib.pyplot as plt
import numpy as np

class MSE_pair:
    def __init__(self, in_model_name, in_mse_value):
        self.MSE_model_name = in_model_name
        self.MSE_value = in_mse_value

    def showMSE(self):
        print( "MSE for ", self.MSE_model_name, " is:[" , self.MSE_value, "]")

def run_classification( preprocess_dict, flags_dict, X_train, X_test, Y_train, Y_test, model_name_in, mse_vec ):
    if( model_name_in == 'Logistic' ):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression( random_state = preprocess_dict['RandState'] )
        classifier.fit( X_train, Y_train )
        Y_pred = classifier.predict( X_test )
        
    elif( model_name_in == "KNN" ):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier( n_neighbors = 5, metric = "minkowski", p = 2 )
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
    
    elif( model_name_in == "SVM" ):
        from sklearn.svm import SVC
        classifier = SVC( kernel = 'rbf', random_state = preprocess_dict['RandState'] )
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        
    elif( model_name_in == "Bayes" ):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        
    elif( model_name_in == "DecisionTree" ):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier( criterion = 'entropy', random_state = preprocess_dict['RandState'] )
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
    
    elif( model_name_in == "RandomForest" ):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier( n_estimators = 10, criterion = 'entropy', random_state = preprocess_dict['RandState'] )
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
                     alpha = 0.75, cmap = ListedColormap(('#FDDA87', '#668AFF')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(['#FFCE58', '#295BFF'])(i), label = j, edgecolors = 'black')
        plt.title( model_name_in )
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()


def run_regression( preprocess_dict, flags_dict, X, Y, model_name_in ):
    if( preprocess_dict["ModelName"] == "Linear" ):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit( X, Y )
    
    elif( preprocess_dict["ModelName"] == "Poly" ):
        from sklearn.preprocessing import PolynomialFeatures
        regressor = PolynomialFeatures( degree = 4 )
        X_poly = regressor.fit_transform( X )
        regressor.fit(X_poly, Y)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit( X_poly, Y )
    
    elif( preprocess_dict['ModelName'] == 'SVM' ):
        from sklearn.svm import SVR
        regressor = SVR( kernel = 'rbf' )
        regressor.fit( X, Y )
        
    elif( preprocess_dict['ModelName'] == 'DecisionTree' ):
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor( random_state = preprocess_dict['RandState'] )
        regressor.fit( X, Y )
    
    elif( preprocess_dict['ModelName'] == 'RandomForest' ):
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor( n_estimators = 20, random_state = preprocess_dict['RandState'] )
        regressor.fit(X, Y)
    
    # Plot graphs
    if( flags_dict["show_graph"] and ( preprocess_dict["ModelName"] == "Linear" or preprocess_dict['ModelName'] == 'SVM' ) ):
        plt.scatter(X, Y, color = 'red')
        plt.plot( X, regressor.predict(X), color = '#FDDA87' )
        plt.plot( X, regressor.predict( X ), color = '#295BFF' )
        plt.title('Salary vs Experience (Test set)')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.show()
    
    elif( flags_dict["show_graph"] and preprocess_dict["ModelName"] == "Poly" ):
        # Visualising the Polynomial Regression results
        plt.scatter(X, Y, color = '#295BFF')
        plt.plot(X, lin_reg_2.predict( regressor.fit_transform(X)), color = '#FDDA87' )
        plt.title('Truth or Bluff (Polynomial Regression)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        plt.show()
    
    elif( flags_dict["show_graph"] and ( preprocess_dict['ModelName'] == 'DecisionTree' or 
        preprocess_dict['ModelName'] == 'RandomForest' ) ):
        # Visualising the SVR results (for higher resolution and smoother curve)
        X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, Y, color = '#295BFF')
        plt.plot(X_grid, regressor.predict(X_grid), color = '#FDDA87')
        plt.title('Truth or Bluff (SVR)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        plt.show()


def run_clustering( preprocess_dict, flags_dict, X, Y ):
    if( preprocess_dict['ClusterName'] == 'KMeans' ):
        # Using the elbow method to find the optimal number of clusters
        from sklearn.cluster import KMeans
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = preprocess_dict["RandState"] )
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        
        # Fitting K-Means to the dataset
        kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = preprocess_dict["RandState"])
        y_kmeans = kmeans.fit_predict(X)
        
        # Visualising the clusters
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
        plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.show()
    
    elif( preprocess_dict['clustername'] == 'Hierarchical' ):
        # Using the dendrogram to find the optimal number of clusters
        import scipy.cluster.hierarchy as sch
        dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show()
        
        # Fitting Hierarchical Clustering to the dataset
        from sklearn.cluster import AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(X)
        
        # Visualising the clusters
        plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
        plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.show()
