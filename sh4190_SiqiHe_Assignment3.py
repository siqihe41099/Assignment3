import nibabel as nib
import scipy.io
from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# import data
labels = scipy.io.loadmat('/Users/siqihe/Desktop/label.mat')['label']

# test
img_test = nib.load('/Users/siqihe/Desktop/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii')

# create a brain mask to extract brain regions
# Here I used an existed function to find the brain mask
# Reference: https://nilearn.github.io/manipulating_images/manipulating_images.html
mask_img_test = compute_epi_mask(img_test)
masked_data_test = apply_mask(img_test, mask_img_test)

# We applied two methods on the data here. One is svm and another one is svm and pca.
# method 1 svm
cv = StratifiedKFold(n_splits=7, random_state=2, shuffle=True)
svc = SVC()
parameters = {'C':[1,5,10],
              'kernel':['linear', 'rbf'],
              'gamma':[0.1,0.2,0.3,0.5,0.8,'scale']}
test_model_select = GridSearchCV(svc, parameters, cv=cv)
test_model_select.fit(masked_data_test, labels)
print('The best parameters of svm are', test_model_select.best_params_)
print('The corresponding accuracy is', test_model_select.best_score_)


# method 2 svm+pca
pca = PCA(n_components=120,random_state=1)
data_reduced_test = pca.fit_transform(masked_data_test)

svc = SVC()
parameters = {'C':[1,5,10],
              'kernel':['linear', 'rbf'],
              'gamma':[0.1,0.2,0.3,0.5,0.8,'scale']}
test_model_select2 = GridSearchCV(svc, parameters, cv=cv)
test_model_select2.fit(data_reduced_test, labels)
print('The best parameters of svm are', test_model_select2.best_params_)
print('The corresponding accuracy is', test_model_select2.best_score_)



# retest
# I repeated the steps above on retest data
img_retest = nib.load('/Users/siqihe/Desktop/sub-01/ses-retest/func/sub-01_ses-retest_task-fingerfootlips_bold.nii')
# create a mask by using compute_epi_mask
mask_img_retest = compute_epi_mask(img_retest)
masked_data_retest = apply_mask(img_retest, mask_img_retest)

# method 1 svm
cv = StratifiedKFold(n_splits=7, random_state=2, shuffle=True)
svc = SVC()
parameters = {'C':[1,5,10],
              'kernel':['linear', 'rbf'],
              'gamma':[0.1,0.2,0.3,0.5,0.8,'scale']}
retest_model_select = GridSearchCV(svc, parameters, cv=cv)
retest_model_select.fit(masked_data_retest, labels)
print('The best parameters of svm are', retest_model_select.best_params_)
print('The corresponding accuracy is', retest_model_select.best_score_)


# method 2 svm+pca
pca = PCA(n_components=120,random_state=1)
data_reduced_retest = pca.fit_transform(masked_data_retest)

svc = SVC()
parameters = {'C':[1,5,10],
              'kernel':['linear', 'rbf'],
              'gamma':[0.1,0.2,0.3,0.5,0.8,'scale']}
retest_model_select2 = GridSearchCV(svc, parameters, cv=cv)
retest_model_select2.fit(data_reduced_retest, labels)
print('The best parameters of svm are', retest_model_select2.best_params_)
print('The corresponding accuracy is', retest_model_select2.best_score_)









