######	Maching-Learning-Pipeline
######	Hridai Trivedy
######	May 2019
######	Hridaitrivedy@gmail.com


A python template for pre-processing data then applying regression, clustering and classification models using sklearn. Post-model accuracy measures, summaries also carried out - all plots done using matplotlib.
Gridsearch also included for hyperparameter tuning.

preprocess_dict selections:
	TrainSplit 		:	Any number between 0 and 1
	ModelType		:	["Classification", "Regression", "Clustering"]
	ModelName		:	["Logistic", "KNN", "SVM", "Bayes", "DecisionTree", "RandomForest", "Poly", "KMeans", "Hierarchical"]
	
Structure:
	1) 	Define file path for reading in by pandas lib
		-	Or define remote repo path online for donwnload [TODO]
	
	2)	Analyse Import
		-	Look at the data types you have brought in
		-	Plot histograms to see the distribution of the data
		-	Find correlations between features
		Important to do this with a view to removing/collapsing down features during the tidying step
		Look at dimensionality reduction techniques [TODO]
		
	3)	Clean data
		-	Remove NAs or replace them
		-	Calculate Medians and replace missing values with these
		-	Class: sklearn "Imputer" [TODO]
			
	4)	Encode the categorical features
	
	5)	Split data into training and test sets
		-	Test stratified sampling [TODO]
		-	Must include a cross-validation step [TODO]
		
	6)	Apply Feature Scaling
	
	7)  Carry out Clustering
	
	8)	Gridsearch to optimize parameters
	
	9)	Run model(s)
	
	10)	plot outputs and accuaries
	
	11)	Repeat from step 8) or revisit technique entirely if sufficient accuary/recall is not met.
	