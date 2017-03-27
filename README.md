Repository for SoftImpute-ALS Python Implementation

=======SoftImpute-ALS=======

*The softImpute.py module is the main source module for this project.
An example of how to run it is in the main routine in that module. This is
reproduced here with explanatory comments on how to interact with the module:

  #initialize the data sets with text_to_CSR function provided
  num_users = 943
  num_items = 1682
  R_train = text_to_CSR('data/ml-100k/ua.base', num_users, num_items)
  R_test = text_to_CSR('data/ml-100k/ua.test', num_users, num_items)

  #initialize the object by giving it the training set and the solution rank k
  sals = SoftImpute_ALS(40, R_train)
  #fit the data set for a given lambda (other options are provided in the fit
  function interface, look at the module for details)
  sals.fit(Lambda=5)
  #compute the RMSE with the computed approximation on the input test data
  print sals.compute_rmse(R_test)

*To test our results, create a 'data' directory in the project root and place
the 'ml-100k' directory from the MovieLens database at:
              https://grouplens.org/datasets/movielens/


*To generate covergence plots, simply set the plots_conv variable to the file
path you'd like the plot to be save, and a convergence plot will be generated.


=======Data Visualization=======

*The module we used for data visualization is dataVis.py, an example of how it
is run is shown in the main routine at the bottom of the file.  The class
'DataVisualization' was imported into our softImpute module to generate
our visualization plots.
