# Multivariate-Linear-Regression
Multivariate Linear Regression using Cross validation.

In this project the used dataset iris.data which has 4 independent attributes and one dependent attribute as our target attribute.
So, as we have more than one independent variables, we have to use multivariate Linear Regression to predict the target variable using Cross validation. 
Firstly, we had read the dataset which had 5 attributes and 150 tuples in it, then we separated the independent i.e. source data and the target class as our X and Y.
After, we had the X, Y data we had to normalize the data as Y had the class labels as string and we had to replace it with integers. 
So, we labelled each Y string with corresponding integer.

 As, our dataset tuples were quite sequentially organized according to their class hence would create a problem when K_folds =3(in cross validation) as it would generate 3 partitions of 50 tuples with each partition belonging to the same class. This would lead to model not learning about the other class label and would miss classify half of the testing data. So, one way to get around this was to shuffle the data.

Cross Validation

Here, I have used K fold Cross validation and implemented a function to do its operation. The function splits the X and Y data in k partitions and then initializes the X_train, Y_train, X_test, Y_test by iterating through the partitions k times and each time assigning k-1 folds to training and one to testing.
As there are multiple independent attributes the equation of prediction of the target calss y would be:

Y = B0  + B1X1 + B2X2 + B3X3 + E

For us to calculate the values of Beta for each fold we will have a matrix of values of beta that is coefficients beta.
We have implemented the model for different values of k like 3,5,7, 10, 12. The values of k are selected such that each time the folds length need not be same, to have a variety in splitting. 
