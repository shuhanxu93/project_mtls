## project_mtls

The final models are located in `./final_predictors/`. The models are designed for protein sequences longer than 25 residues. First, you need to move your own fasta and PSSM files into `./final_predictors/` to predict the secondary structures of your own proteins. Then use final_seq.py to predict from primary sequence or final_fm.py to predict from PSSM. Give the input filename and output filename in the terminal as arguments when running either of the scripts. The result is printed to the screen and saved to the output file in the same directory.

Here is a demonstration using the provided fasta file in the directory:
```
cd final_predictors
python3 final_seq.py example.fasta output_fasta.txt
```

Here is a demonstration using the provided PSSM file in the directory:
```
cd final_predictors
python3 final_fm.py example.pssm output_pssm.txt
```

To view my scripts for cross_validation, training and scoring, please go to `./scripts/`.

Cross-validation scripts: `gridsearch_#inputtype_#method_#classweight.py`

Training scripts: `train_#inputtype.py` (need to modify window size, model name and hyper-parameters)

Scoring scripts: `score_#inputtype_looper.py` (for test set)
                 `score_#inputtype_looper_newproteins.py` (for 50 new proteins)

To view the results of cross_validation, please go to `./results/#inputtype_#method`. Each folder contains the cross-validation results for each type of model.

To view the results of scoring, please go to `./results/reports/`.

To view the dataset, psiblast, pssm, 60 new proteins, pdb and dssp files, please go to `./datasets/`.

To view the bash scripts, please go to `./bash/`.

Bash script for psiblast: `psi_blast.sh`.

Bash script for DSSP annotation: `dssp.sh`.

The 16 models trained as mentioned in the report are not pushed into this repository as some of them are too big.

### 20180316

Today was a pretty awesome day. I managed to figured out with my friends how to solve some of the problems that I was facing. One issue was reduction of 8 states to 3 states. The dataset that I was given contained only 'H', 'E' and 'C'. There was no information how the three states were defined. John and Marco also did not provide me with much information. Hence, I was not sure how to process the 8 states generated from DSSP. Maryia then told me that I could search the dataset on PDB to find out the original 8 states and then figure out the rule for states reduction. Basically, the rule used by my dataset was converting all classes that are not 'H' and 'E' to 'C'. Now I can make fair evaluation of my predictor using the new proteins. By the way, I realised that the first and last amino acids in my dataset may not be the N and C terminal amino acids. Because residues at the terminals are usually very floppy, 3-D structural determination methods may not be able to obtain the coordinates of the terminal residues correctly. Hence, the protein that we are given are most likely truncated. This makes me question the paddings we add to the ends. Before we add the paddings, should we enlongate the sequence to its true ends and appends 'C's to the secondary structure since the terminal residues are flexible?

I have made more progress in my scripts. Continuing from what I have left off on Wednesday, I finished the scoring script for models trained on PSSM. I have also included in script the code for plotting confusion matrix which I took from scikit-learn. In addition, I have modified my contour plots so that it displays all 7 figures in one figure. Finally, before I leave the lab, I downloaded the psiblast pssm for the new proteins so that I can test them at home. For tomorrow, I will generate plots for CV and scoring in the morning and write report in the afternoon.

### 20180315

The original plan for today is to get all my scores and plots ready so that I can start writing report on Friday. However, I hit a roadblock when I realise that the last task of getting 50 new proteins is not that easy as it seems. So I spend almost the whole day trying to figure out how to find good protein structures from CullPDB, retrieve good pdb files from pdb, generate secondary structures using DSSP and finally, retrieve sequence and secondary structures from DSSP. I was pretty stressed out the whole but thanks to the guidance from my peers, especially Helen, I managed to get all the codes working. I may not have written the best code I could for this task but at least I managed to get things working. I hope that in the near future, I can revisit some of the codes that I have written and make them more robust. Here are two cool things I have learnt today: 1. Using regular expression to substitute lower case character for another character. This is handy when the cysteines are substituted with lower cased alphabets which indicates that they are forming disulfide bridges. 2. Specifying the column at character level in np.genfromtxt(). This is useful if one wants to strictly reads from one straight column. For tomorrow, I will be generating my scores and plots. Hope that everything goes smoothly tomorrow.


### 20180314

I have finished quite a lot of tasks today. I have rewrote scripts to train the sequence models. I have created two dummy models to test my other scripts. David taught me how to use dictionary to save the model together with the window size so that I do not have to hard code the window size when I am using the model. I have writen the scoring script which includes accuracy, confusion matrix, recall, precision, f1 score, f1 macro, mcc and q3. I added a loop which runs through all the models and store the scores in a csv. Since the arrays are converted to strings when they are loaded from csv, I have to store each element separately. However, I cannot use this method for confusion matrix. I think I will deal with confusion matrix separately. I have also written a script for plotting the best accuracy scores, including error bars, for each window size. Using this plot I have analysed my pssm grid search. It turns out that the optimum window size for my pssm is 11(`class_weight='balanced`) and the score deceases from 11 onwards. See the figure below. I expected the optimum window size to be at least 15. I need to ask the group about this result on Friday. I will continue the momentum tomrrow.

![pssm.png](/figures/pssm.png)



### 20180313

I continued to run the gridsearch today. Warnings for F1 macro kept popping out during the cross-validaty. I realised that this happened when there was no correct prediction for at least one of the classes. As a result, F1 score for that class is undefined and set to zero.

Hope that I will make more progress tomorrow. 


### 20180312

#### PSIPRED
Although the PSIPRED paper is easy to understand, the methods is not thoroughly explained. One thing that stood out the most is how the author encoded the features. In the paper, the author states that the input layer consists of "315 input units, divided into 15 groups of 21 units". 15 represents the sliding window size. Instead of 20 units per amino acid, they encode one extra unit to indicate the N or C terminus. On Friday, John recommended using only 20 units per amino acid as having an extra unit for padding would means that an artifical amino exists and is given the same weight as a natural amino acid and this does not make biological sense.

Lets leave that for now. There is another issue which I managed to figure out. For the substitution matrix, every element is a log odds ratio. For the padding, all log odds ratios should be negative infinity. Instead of appending rows of negative infinity to the substitution matrix, I can append rows of zeros after I have scaled the substitution matrix with logistic scaling. Marco recommended not using the substitution matrix as the author did not use the logistic scaling exactly as they have written in the paper. Marco said it is difficult to understand exactly what the authors did in the script. Hence, I will not use the substitution matrix in my predictor.

#### Uniref50
I reran the psi-blast using Uniref50 out of 2 reasons. First since swiss prot database does not have contain as many sequences as Uniref50, frequency matrices returned from swiss prot does not capture much information. Many of the elements in the frequency matrices are either 100 or 0. I believe that they will make poor models. Second, psiblast did not return any pssm for 1wfbb-1-AUTO.1.fasta. This sequence is short and contains long stretches of 'A'. Hence I want to see if using Uniref50 will return me a pssm for this sequence.

I have set my evalue to 0.001 since the database is much larger. However, using Uniref50 still does not give me a pssm for 1wfbb-1-AUTO.1.fasta. Marco said that if a sequence is poor, psi-blast will not return me pssm even if I use Uniref90. Hence, I will remove the sequence from my datasets.

#### Contour Plot
I managed to get my new contour plots after 2 days of non-stop data crunching. This time the contour plots represent cross-validation accuracy of models train using 20 units/amino acid encoding instead of 21. Here are the results:

window size 11
![win11.png](/figures/win11.png)

window size 17
![win17.png](/figures/win17.png)

window size 23
![win23.png](/figures/win23.png)

What I see from these figures is that as window size increases from 11 to 17, the highest accuracy score increases. However, if the window size contines to increase to 23, the highest accuracy score decreases. This may due to fact that larger window size also adds more noise to the input, making the prediction less accurate. This highest accuracy scores for balanced and non-balanced class-weights are similar. I might need to use confusion matrix and f1 score to see which one is the better predictor.

#### Train PSSM fm using gridsearch
I have tested the PSSM parser I wrote on Sunday. Using the default rbf kernel parameters give me ~4% higher accuracy compared to the primary sequence predictor. I will run grid search over the next few days to find the optimum parameters.


### 20180311

### 20180309

```
Hi John, David and Marco, for this week's submission:

The file ./scripts/train.py trains the model using window size of 17 and save the model as ./models/model.pkl.
This step has already been done.

The file ./scripts/predict.py predicts the secondary structures of the proteins in ./datasets/test.fasta using window size of 17 and outputs the results on screen and also in the fasta file ./outputs/predictions.fasta.
```


### 20180308

Today was a fruitful day.
I have not rested.
My computer has not rested.


### 20180307

Yesterday was a stressful day for me. I was puzzling ove the simplest and most logical way to train and cross-validate my predictor. Hence, I consulted the TAs at the earliest time possible today. Marco and David are extremely helpful with my queries (John was busy with other students) and the following points were what I understand:

1. GroupKfold

2. I do not necessarily need Q3 for cross validation

Here is what I have done today:


### 20180306

Project keeps me up at night.

What is the different:

protein seq --> window fragment --> shuffle -->split into kfold equally

protein seq --> shuffle --> split equally --> window fragment

Still feel puzzled about this.

I think my explaination from previous post is wrong.

Only advantage is that you can do Q3.

What is the advantage of using Q3 for cross validation?

### 20180305

What I like about this course is that there is no clear set of instructions laid out for me. We are required to read to expand our knowledge and we are encouraged to experiment (although this can be time consuming sometimes). Although I am confident of the steps to take at the start of the course, I am continuously challenged to think otherwise during the journey. Each article I read makes me question my method. This uncertainty definitely makes me feel uncomfortable but I know that this will make me a better thinker in the end. It is as if I am traversing along a stochiastic gradient descent (sam would love this). There is no guarantee that each step will leap you to the optimal minimum but you know that you will land somewhere close eventually.

Another thing that I am feeling thankful for is this journal. I can spend hours and hours in front of the computer making no headway. And the longer I stare at the screen, the slower my brains think. Eventually, my thoughts will come to a halt. Then, I will come to this diary, list out all my frustrations and sort them out like a garbage man digging through the trash. Perhaps I can find bits and pieces of ideas that are still useful.

So I was rewriting my code, almost from scratch. This is because the cross-validation function of scikit learn only evaluate a predictor on the level of a window. However, most other research performs cross-validation on the level of a whole protein sequences which makes sense. First of all, the end goal of the predictor is to predict the secondary structure of the whole protein correctly, and not to predict only each residue correctly (although predicting each individual residues correctly will correlate with predicting the whole protein correctly). More importantly, as John mentioned, if we only scikit learn with peptide window, the predictor will be biased towards longer proteins since longer proteins will have more peptide window. The predictor should treat every protein equally.

At the same time, I am writing the code that treats the boundary cases of proteins as special windows. Although I still do not fully comprehend the benefit of doing this, I think it might be better for me to test out different methods to expand my repertoire of coding tricks and techniques.

`import this` The zen of python captures some of the essence of python. It is philosophical in nature but there are some ideas inside that we can all agree upon. Currently my zen level is still 15%, i.e. I can only appreciate 15 percent of the ideas.


### 20180302

```
Hi John, David and Marco, the file for today's submission is week_3_report.py under the scripts sub-directory.
```

I have listened to another two amazing presentations during journal club today and learnt a lot today.

David first presented on the topic of transfer learning. It is the concept of borrowing a trained neural network, or any other trained estimator, for a specific task and make minor changes to it so that it can perform a different but similar task. David gave the following example. If we have trained a convolutionary neural network to distinguish a cat from a human, do we need to retrain the neural network from scratch so that it can distinguish an armadillo from a platypus? Since the bulk of the network is extracting features from an image such as applying filters, that part of the model can be retained and it can perform just as well on armadillo and platypus images as on cat and human images. However, we still need to provide a smaller training set of armadillo and platypus images to retrain the decision making module of the network so that it can classify different things. This raises the idea of building a knowledge-base neural network which is a general purpose CNN trained from a giangantic dataset of high quality training examples. The user only need to feed a much smaller training set to traing the CNN to perform more specific tasks. David then bring the topic back to biology. He said phi angles, psi angles and solvent accessible surface area are highly correlated characteristics of a protein. We can use a set of protein primary sequence to train the estimator to predict phi angles, we can use another smaller training set to tweak the estimator to predict psi angle or solvent accesible surface area. Transfer learning seems to be very suited for sequence labeling probelems, such as predicting secondary structure, solvent accessibility, contact number from protein primary sequence.

Next, John talk about a case study of using machine learning to identify individual whale from photos. [Here](https://blog.deepsense.ai/deep-learning-right-whale-recognition-kaggle/) is the link. It was an competition on [Kaggle](https://www.kaggle.com/competitions), where people can submit their machine learning models to solve real world problems. Below are some of the key ideas I have taken from this case study.

#### Domain knowledge is important to solving machine learning problems.
In the competition, the winning team has no prior knowledge on how to identify a particular whale from image. The only knowledge they get from wikipedia and the organizer is that the white patch on the whale head can be used to distinguished one specimen from another. And they have used it to great success. Their method focuses on capturing this piece of information from each photo and use it to train their model. This also reminds me of the lecture that Gunnar gave us on membrane protein topology prediction. He mentioned that the positive-inside rule not only tell us the orientation of the transmembrane protein but it can aid predictors in determining the number of transmembrane helices. If the predictor has both a low and high strigency threshold, it can use the positive inside rule as a jury to determine which model produce by the predictor is the more reasonable one.

#### Break down a difficult task into many smaller and more managable tasks.
Although this may take more manhours but the results is worth it. Training a single CNN to take in a raw image and categorise the whale correctly is difficult. What the team did is to train several CNNs to perform this task. They first have a CNN that locates the head of the whale. Then they have another CNN which orientates the head and produce passport-like photos. Then they have a final CNN which take this preprosessed photo and categorise the whale. While a CNN may perform the whole task well, it can achieve great accuracy for a sub-task. By combining these CNNs together, they can have a great overall performance.

#### Machine learning is still more of an art than science.
A successful machine learning practitioner does not just devote all his or her time on a single idea. The best way approach is still to test out multiple ideas and learn as much as one can in a short period of time. In this case study, there are many instances in which the authors realise that a method is good only after they have tested it out. In essence, you can never be absolutely sure whether your idea is a brilliant or poor until you have tested it out.


### 20180301

Was feeling a bit ambitious today. I was planning to perform a grid search for optimum pair of C and gamma for my predictor of window_size=17. Previously I face the issue of SVC not converging for large value of C and hugh training set size. The fact is SVC learning will always converge and it is taking a very long time for that case. I then learnt from StackExchange that if I increase the `cache_size` of SVC, I could reduce the amount of recomputation and speed up the learning. Even so, it is still computationally expensive to perform brute force search for C(2<sup>-5</sup> to 2<sup>15</sup>) and gamma(2<sup>-15</sup> to 2<sup>3</sup>) as recommended by Hsu *et al*<sup>1</sup>.

While waiting for the search to complete, I worked on my presentation on Deep Convolutional Neural Fields by Wang *et al*<sup>2</sup>. The paper discusses some of the novel ideas in the deep convolutionary neural network field. It is simply amazing how the predictor can achieve such a high accuracy with a simple architecture and a small window size(11). I am hyped for the presentation next week.

1. Hsu, Chih-Wei, Chih-Chung Chang, and Chih-Jen Lin. "A practical guide to support vector classification." (2003): 1-16.

2. Wang, Sheng, et al. "Protein secondary structure prediction using deep convolutional neural fields." Scientific reports 6 (2016): 18962.


### 20180228

Finally, I have moved my diary to my github project_mtls project page. This has ended the trouble of typing in a .txt file which does not allow one line to be spilled over to the next line if it is too long. Furthermore, I can put links and figures in my diary and this makes referencing much easier.

I worked on the issue of shuffling and splitting my training and test set. The idea of having a separate test set is to evaluate the performance of the predictor on data that it has not seen before. This means that neither the weights nor the hyper-parameters should be learnt from the test set. In other words, values of the weights and hyper-parameters need to be decided based on the information from the training set only. Consequently, I need to make sure that I am always using the same training set when playing with the hyper-parameters. Although I can use a seed to ensure that the `train_test_split` always generate the same training set, this approach may not work if I want to change window size. This is because the array that I feed into the train_test_split will have a different shape and different values after I change to another window size and I cannot guarantee that it will always be split the same way with the same seed. At least I have not tested this notion yet. Hence, instead of using the in-built `train_test_split`, I implemeted my own code to 'randomly' split the dataset into training and test sets. Using `seed=0`, I first permutate a numpy array of integers ranging from 0 to n_samples - 1 and slice the first 60% of the array to index the training set. The remaining 40% of the array is used to index the test set. Although this method is not asthetically please, it ensures that I will always get the same training set no matter how I adjust the `window_size`.

### 20180227

Today, I have spent most of my coding time reading documentation. C and gamma are called hyper-parameters because they cannot be learnt by the estimator. What can be learnt are the weights or theta. Scikit learn has several inbuilt classes which can search for the optimum set of parameters. One of which is the exhaustive grid search. Users just need to specify the range of each of the parameters and put them in a dictionary. The `GridSearchCV` can take this dictionary and calculate the cross-validation score of every combinations of the parameters listed in the dictionary. In addition to the exhaustive grid search, Scikit learn also has a randomised search approach called `RandomisedSearchCV` which can search for the parameters from a distribution of values. Calling the `cv_results_` attribute will return a dictionary of the results which can then be fed into pandas to produce a dataframe. This dataframe can then be used to make contour plots of the C and gamma grid of cross-validation scores. One can also specify `error_score=np.NaN` in case of failures so that the search will be complete.

Beyond reading documentations, I have watched a [youtube video](https://youtu.be/MOdlp1d0PNA) which teaches me the steps to get better in machine learning. Zach Miller mentioned two books in the video which worth reading. One is 'Data Science from Scatch' and another is 'Machine Learning with R'. I am definitely going to grab the first book to read. He also thanked the website [Machine Learning Mastery](https://machinelearningmastery.com/) which taught him a lot. Watching this video makes me feel that I am never alone in my journey to learn machine learning and data science. There will alway be countless of teachers whom I will meet along the way. 


### 20180226

Today I have plotted the learning curve for my first predictor. I used a window size=17, C=1.0 and gamma=0.003 for the plot. I choose that window size because it is the one used by Jnet and GOR method. C is 1.0 which is the default value used in SVC. Gamma is 0.003 which is approxmiately 1/n_features (1/357), the value used in SVC under default settings. The purpose of plotting a learning curve is to diagnoise whether the predictor is suffering from high bias or high variance.

With few training examples, the training score(accuracy) is high(close to 1) and the cross validation score is low. This is because the predictor is able to fit the training set well but unable to generalise to fit the cross-validation set. As the number of training examples increases, the training score will decrease because it becomes more difficult for the predictor to fit so many training examples well. On the other hand, the cross validation score increases as the predictor is able to produce a better model which can generalise well to the cross-validation set.

In an ideal predictor, the training score will remain high while the cross validation score will be get very close to the training score as the number of training examples increases. If the predictor is suffering from high bias, both the training score and the cross-validation score will be low and they are close to each other when the predictor is trained on large number of training examples. This is because the model is not complex enough to fit either the training or the cross validation set. On the other hand, if the predictor is suffering from high variance, the training score will be high but the cross-validation score will be low because while the model is complex enough to fit the training set, it cannot generalise well to fit the cross-validation set.

Here is my learning curve:
![learning_curve2.png](/figures/learning_curve2.png)

I started the plot from less than 1000 training examples. If I use even few examples, one will see the training score dropping from 1 and the cross-validation rising from zero. What is more important is how the curve appears as training size increases. It seems that both the training score and cross-validation scores are flattening out at 0.62. Since both scores are quite low and close to each other, our model seesms to be suffering from high bias. What I will spend the next few days is to tweak the C and gamma hyper-parameters. If I have more time, I will also try out a range of window sizes.

On a side note, John mentioned that it is best practice to treat peripheral windows and internal windows separately. He said that it will be easier for downstream implementations. As I do not have as much intuition and experience as him, I cannot forsee the problem that I will face. I think I will stick with my code for now which is simpler. If I face another issues in the future, it will be a stronger lesson for me. Anyway, it will not be that difficult to change the code since it is much modular now.


### 20180224

Spend most of my time reading and understanding sklearn SVM today.

Although I have some previous understanding of SVM from Professor Andrew Ng's Machine Learning coursera course, it not enough not appreciate the intricate details of SVM. Let me do a short recap of SVM. The mathematical formulation of SVM's cost function consist of an error term, i.e. how much the predicted y-value differs from the actual y-value, and an regularization term, i.e. the sum of squares of the weights. There are two parameters to adjust for a SVM running on a RBF kernal. The first is C which controls how much the error term contributes to the cost function with respect to the regularization term. The larger the value of C, the greater the contribution of the error term and hence, the higher tendency of the model towards overfitting or high variance. The second parameter is gamma which is specific to the RBF kernel. In Professor Ng's video, he used 1/(2sigma) instead but the idea is the same. The kernel gives us the similarity scores of an input vector with respect to each of the training example. In the case of a RBF kernel, the larger the value of gamma, the closer to a training example the input vector needs to be in order for the similarity score to be high. Hence in a way, the greater the value of gamma, the more complex the shape of the decision boundary and hence the greater the tendency for the model to overfit.

However, there are still many inner workings of SVM that I do not yet understand. Unlike logistic regression or neural network, SVM does not calculate the probability of an example to belong to a class. Instead, SVM uses the score produced by the decision function to determine the class of an example. This decision function uses a subset of training examples (also known as support vectors) to calculate the score. This is only of the computational tricks SVM uses to speed up the calculation. Besides this, SVM can also implement an "one vs one" (OVO) approach  to mutlticlass classification. While the traditional "one vs rest" (OVR) approach trains n_class classifiers, the OVO trains <sup>n_class</sup>C<sub>2</sub> classifiers. I do not understand the benefit of OVO approach since it is more computationally expensive. This is a topic I will explore in the future.

Beyond that, the SVC classifier from sklearn enables user to set weight for each class under the class_weight keyword argument. This changes the C parameter for each class. This is particularly useful when the examples of the training set are not evenly distributed among the different class labels. Hence, setting the class_weight can reduced the tendency of the SVM to be biases towards the more populated class. I think it might be worthwhile to try out this parameter since for my protein secondary structure predictor, more than half of the training examples belongs to random coil and for the other half of the examples, there are more alpha helices than beta sheets. Another approach to this problem is to use the same number of random coils, alpha helixs and beta sheets in the training set.

Lastly, I found out that in addition to the inbuild k-fold cross validation iterator, there is also a stratified k-fold cross validation iterator. Stratified k-fold is superior in a sense that it ensures the proportion of the class labels in the training set is preserved in the k-1 training folds and the one cross-valdation fold. This is especially useful for datasets with skewed classes.

For coding today, I create a training sets called cas3_removed.txt which contains all the examples in cas3.3line.txt except sequences which contain 'B', 'X' and 'Z'. I think the ambiguous cases will reduce the accuracy of my predictor. I will explore strategies to deal with ambiguous cases later in the project. One thing I noted today is that training does not converge when I use all training examples with window_size = 17 and C = 1000. I might need to set a limit for the number of iternations when I am doing the search for optimal pair of C and gamma.


### 20180223

Participated in the first journal club of the course. Marco said it was specially designed for us.

David started off the session with a introduction into deep learning. He began with the history of machine learning from curve fitting to expert systems to feature engineering. Then he spent some time discussing deep learning. What is amazing is that some of the layers of a convolutionary neural network look very similar to the gabor filters used in feature engineering. It is amazing to think about how a machine can learn this from scratch without human intuitions. Although I do not fully understand David's presentation, he gave us website where we can learn deep learning on ourselve such as this one: [Udacity](https://eu.udacity.com/course/deep-learning--ud730).

The second presentation was on visualising convolutionary neural network(CNN) by another member of the lab. It was based on the 2013 paper by Matthew D. Zeiler and Rob Fergus. Many of the ideas in the paper are still alien to me but the basic idea is to develop methods to visualise the hidden layers of the CNN and use that insight to understand how CNN learns. What is fascinating is that in a separate experiment, the authors use a gray square to cover part of the image and test if the computer can still categorise the image correctly. In one instance, they tried to cover different parts of a image of a pomeranian. While the computer can still recoginise the pomeranian when the legs or ear are covered, it is unable to do so when the face of the creature of covered. In a way, this is very similar to the way human see and recognise an object. We do not pay attention to all parts of an object. We only notice the distinctive features of that object. The presenter then gave us this question to ponder: If we can identify the most salient features used by a computer in categorising a image or an amino acid, can we transfer that knowledge back to human intuition? Suppose a computer can use a certain set of features of a protein sequence to correctly categorise the secondary structure of each amino acid, can we learn from that set of features and understand the physics and chemistry of why a particular amino acid of a protein sequence adopts a particular secondary structure? In a way this is similar to reverse engineering. Maybe we shall call this reverse science.

After the journal club and group meeting, I talked to John about the possible problems I face in creating the secondary structure predictor. Below is a summary.

* Qn1: What metric should we use to optimise our predictor? Accurarcy score? Matthew correlation coefficient? Q3?
* Ans1: Pick one but you must justify why you use that metric.

* Qn2: How shall we optimise 3 parameters such as C, gamma and window size? Is it too time consuming to use try all combinations.
* Ans2: One can do brute force optimisation by trying out all combination of C, gamma and window size. It is preferable to use a more heuristic approach. You can first try test out the different window size and narrow down the range of the range of the window size. Then you can perform a grid search of C and gamma over that small range of window size.

At night, I have reworked my script to make the structure clearer. The functions are much shorter and more modular. I also included a main function to run all the sub functions. Although the performance and the result is the same, I think it is easier for me to improve it later.

I shall continue tomorrow.


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
* Predictor had an accuracy of ~60% (training set score) with 11000 training examples, window size = 17 and SVM default settings.
* Predictor had an accuracy of ~99% (training set score) with 11000 training examples, window size = 17 and C = 1000.
It seems that my inputs are processed correctly and SVM is learning from my training set. There is still a long way to go.

Tasks for tomorrow:

Write script for cross-validation.
Plot learning curve for predictor to evaluate bias vs variance.


### 20180219

Tasks done today:

Pushed my repository online.
Read the remaining github software carpentry tutorials.

Tasks for tomorrow:

Read first half of python software carpentry tutorials.


