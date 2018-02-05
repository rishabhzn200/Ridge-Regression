Dataset: linRegData.npy
The data is a matrix (100, 2). Column 1 is x and Column 2 is y.

A polynomial of degree 15 is used to fit the data using ridge regression. i.e. x is converted to [1, x, x2 , x3 , . . . , x15 ]' .
5-fold cross validation is used to estimate the best learning rate = λ from the set, λs = [0.01; 0.05; 0.1; 0.5; 1.0; 5; 10];
Best λ is selected and θ is estimated.
Outputs: Plot of cross validation error and train error vs lambdas and plot of polynomial fit to the data.
