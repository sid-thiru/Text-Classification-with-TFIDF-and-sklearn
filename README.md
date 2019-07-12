# EXPERIMENTS WITH SKLEARN

### DATA PRE-PROCESSING
* Each product is represented by its 'Title' and 'Description'. There are 209 classes into which a given product can be classified. This is not a multilabel classification problem, so each product has to be assigned one out of the 209 classes only
* The data was cleaned by removing stopwords, punctuations and special characters from the text
	
	
### FEATURE EXTRACTION
* Each product is represented by a document, which is it's Title and Description combined
* The cleaned up data is represented as TFIDF vectors
* Only unigram tokens are a part of the vocabulary, as it was observed that n-gram tokens led to a slight drop in classification accuracy 
* Limiting the vocabulary size by setting parameters such as max_df (frequency of word occurrence across the dataset) further boosts classifier performance by a small margin
* Dimensionality reduction through PCA (Truncated SVD in this case, because TFIDF features are sparse) dropped the precision by a few points, and so it was decided to retain the original dimensionality of the TFIDF vectors


### CLASSIFIERS
* Naive Bayes
	* Multinomial Naive Bayes serves as a quick baseline, achieving 75% precision with very low training time
	
* SVM	
	* The classifier used here is a linear SVM, as it was observed that an RBF or polynomial kernel doesn't perform well
	* The classifier achieves 83% precision
	* While this is the best performing classifier, it also requires the longest training and prediction time among the ones listed here

* SGD
	* This classifier achieves 81% precision, but at the same time is a lot faster than the SVM
	* The optimal loss function is 'modified huber loss'
	
* Multilayer Perceptron
	* This classifer achieves 79% precision, when optimized using LBFGS
	* The MLP might perform better than the SVM as data size grows

* Voting Classifier
	* Linear SVM and SGD are used to provide two votes, and both the votes are given equal weights
	* This classifier achieves 82% accuracy, and takes a longer time to train than any individual classifier

* Random Forests
	* This classifier achieves 75% precision at best
	* It also requires high training time