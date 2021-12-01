# sh4190_SiqiHe_Assignment3

- How to run your code? Any variable needs to be changed before we run your code (e.g., variable for directory)? \
The code I submitted is python file and I changed the file python from absolute python to relative python. So you may not need to revise anything before you run my code.

- Describe the results of the experiment in your own words. Compare the results of two approaches (e.g., SVM only vs PCA+SVM) and briefly discuss  why one works better/worse than the other.  \

For test image, I tried two approaches: SVM only vs PCA+SVM. For cross-validation, I used stratifiedkfold, since stratified make sure that each fold of dataset has the same proportion of observations with a given label. I split data into 7 sets and fix the random state to ensure that the results are reproducible. \
SVM only: As for svc parameters, I use GridSearchCV() to try different combination of parameters in svc, such as kernel and gamma. The best parameters are 'C': 1, 'gamma': 0.1, and 'kernel': 'linear' and the corresponding accuracy is 0.923. \
PCA+SVM: In this method, I use pca to do dimension reduction. I adjust the n_components parameter for pca() several times and find out the best parameter is 120 here. As for svc parameters, I use GridSearchCV() to try different combination of parameters in svc. The best parameters are 'C': 1, 'gamma': 0.1, and 'kernel': 'linear' and the corresponding accuracy is 0.881. \
I think the results why pca worsen the accuracy is that using pca sometimes can lose some spatial information which is important for classification. However, both two results are larger than 0.85. \

For retest image, I repeated the steps above and usedtried two approaches: SVM only vs PCA+SVM. For cross-validation, I used stratified 7-fold. 
SVM only: As for svc parameters, I use GridSearchCV() to try different combination of parameters in svc, such as kernel and gamma. The best parameters are 'C': 1, 'gamma': 0.1, and 'kernel': 'linear' and the corresponding accuracy is 0.809.  \
PCA+SVM: In this method, I use pca to do dimension reduction. I adjust the n_components parameter for pca() several times and find out the best parameter is 120 here. As for svc parameters, I use GridSearchCV() to try different combination of parameters in svc. The best parameters are 'C': 1, 'gamma': 0.1, and 'kernel': 'linear' and the corresponding accuracy is 0.870.  \
I think the results why pca make the accuracy better in this case is that using pca reduce the number of input variables which result in a simpler predictive model that may have better performance. The advantage is to reduce the difficulty of computation. \
In conclusion, using pca should not definitely make the results better or worse. In theory, it should make no difference. \

- Briefly discuss the limitation(s). How can you improve it? 
1. I did not use realign or segment this time, so I am not sure whether the results will be better if I apply realign and segment on the data. Next time, I may try it on matlab.
2. This time, I use compute_epi_mask to find the mask. But I cannot determine whether this is the best mask. I think I can create mask manully instead of using existed function. And I can compare the results by using different masks.
3. I fixed the random state. However, I found out that for some random states, the results are lower than 0.85. I haven't had an idea to solve the problem.
