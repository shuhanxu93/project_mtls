## project_mtls

### 20180222


I am planning to move my diary to github page during the weekends so that I can put figures in my diary. This will allow me to better track my progress.

I have spent most of the day readying python and SVM tutorials and gain valuable insights to creating a predictor and writer better code in general. The last 3 tutorials on python software carpentry have been particular instructive and worth rereading in the near future. They focus on debugging and building pipelines. A recurring theme in the tutorials is the importance of using assertion statements. The general rule is to include assertion statements in the beginning and end of a function. This serves to ensure that the input is safe for the function and the output is correct. Another skill I learnt is to apply a technique called defensive programming. By writing a series of tests - assertion statements - before writing the function, I will not be biased towards testing the 'correctness' of my function and I will have a better idea of what my function is suppose to do. The tutorial on command-line programs has given me a lot of insights to creating more readable and better structured code. One of the most essential ideas is to write many short functions instead of a long one. A rule of thumb is a dozen line for each function. In the beginning or end of the script, one should write a 'main()' function to chain all the short functions together. Writing short functions makes it easier to test and debug. I will be implementing these ideas in my code in the coming days. There are other important lessons about building pipelines which will be essential during the later stage of my project. I will revisit this tutorial later.

In addition to python, I read the primer on organising computational biology projects. While I can understand the gist of the paper, I would need more computational experience to appreciate the technical details of the paper. Thus, I decide to create as simple a project folder structure as possible for my purpose. What is most essential right now is the ease of organising my scripts and data.

Lastly, I would like to talk about the paper "A Practical Guide to Support Vector Classification"(Chih-Wei Hsu et al.). It is a very succinct introduction to SVM. It avoided the mathematical details to SVM but conveyed some of the most important practical knowledge of SVM for beginners. I will highlight the procedure of optimising C and gamma parameters here. Find suitable values of C and gamma may produce great improvement in the accuracy of the predictor. The author recommends using a grid search method to find the best pair of C and gamma. In addition, they suggests first testing a corse grid of C and gamma before using a fine grid on a smaller region of C and gamma. This will save valuable computational time. Making a contour plot of C and gamma cross validation accuracy is ideal for selecting the range of C and gamma for fine grid.

Finally I would like to add some thoughts about my project. I decide first to not consider the examples with ambiguous amino acids. One thing that puzzles me is the Q3 score. Each example in my dataset is only a window size fragment of a larger protein. Should I optimise my predictor base on accuracy of predicting these examples or the Q3 score of predicting the whole protein? How about reporting the accuracy of my predictor? Should it be based on Q3 score as well? Finally, while optimising 2 parameters may be simple using the grid method, optimising 3 parameters (window size, C and gamma) may not be so trivial. I need to read more about optimisation methods in the weekends.


### 20180221

Tasks done today:

Finished first half of python software carpentry tutorials.

Didn't do coding today as I was tutoring a friend on dynamic programming.


### 20180220

Tasks done today:

Completed parser and data preprocessing.
Predictor had an accuracy of ~60% (predicting training set) with window size = 17 and SVM default settings.
Predictor had an accuracy of ~99% (predicting training set) with window size = 17 and C = 1000.

Tasks for tomorrow:

Write script for cross-validation.
Plot learning curve for predictor to evaluate bias vs variance.


### 20180219

Tasks done today:

Pushed my repository online.
Read the remaining github software carpentry tutorials.

Tasks for tomorrow:

Read first half of python software carpentry tutorials.


