# dmg
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# What: Weight lifting exercise classification problem
# Who: Dana M George
# Purpose: 
#   clean data
#   feature engineer
#   choose best model, train and test
#   accurately predict exercise done by participant

# In[209]:


# Load in notebook dependencies
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib import cm
from datetime import datetime
pd.set_option('display.max_rows', 500)


# In[186]:


# Load in Data
df_raw = pd.read_excel(r'challenge_data_set.xlsx')


# ------------------------ START FEATURE ENGINEERING ------------------------

# In[187]:


df = df_raw.copy()
# remove variables with all null values
df = df.dropna(axis=1, how='all')

# Delete columns containing >= 75% nulls
perc = 75.0
min_count =  int(((100-perc)/100)*df.shape[0] + 1)
df = df.dropna(axis=1, 
                thresh=min_count)

# remove timestamps and names
y = df[['classe']]
df = df.drop(['classe', 'Unnamed: 0', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window','num_window'], axis=1)
# check data types and nulls
df.info(verbose=True, null_counts=True)  # now all variables have the same # of data points


# In[189]:


# check target variable
y.head()


# In[190]:


# Look at features, distributions, skew
x = df.copy() # keep save state of df
x.plot(kind='box', subplots=True, layout=(50,3), sharex=False, sharey=False, figsize=(10,100))
plt.savefig('x_boxplot')
plt.show()


# In[191]:


# More Distribrutions
import pylab as pl
x.hist(bins=30, figsize=(50,50))
plt.show()

# some normal distributions
# some similar distributions, possible correlation
# some skewed data


# In[134]:


# # Matrix
# features = ['roll_belt','pitch_belt','yaw_belt','total_accel_belt','kurtosis_roll_belt','kurtosis_picth_belt','kurtosis_yaw_belt','skewness_roll_belt','skewness_roll_belt.1','skewness_yaw_belt','max_roll_belt', 'max_picth_belt', 'max_yaw_belt', 'min_roll_belt', 'min_pitch_belt', 'min_yaw_belt', 'amplitude_roll_belt', 'amplitude_pitch_belt', 'amplitude_yaw_belt', 'var_total_accel_belt', 'avg_roll_belt', 'stddev_roll_belt', 'var_roll_belt', 'avg_pitch_belt', 'stddev_pitch_belt', 'var_pitch_belt', 'avg_yaw_belt', 'stddev_yaw_belt', 'var_yaw_belt',  'accel_belt_x', 'accel_belt_y', 'accel_belt_z', 'magnet_belt_x', 'magnet_belt_y', 'magnet_belt_z', 'total_accel_arm', 'var_accel_arm', 'avg_roll_arm', 'stddev_roll_arm', 'var_roll_arm', 'avg_pitch_arm', 'stddev_pitch_arm', 'var_pitch_arm', 'avg_yaw_arm', 'stddev_yaw_arm', 'var_yaw_arm', 'gyros_arm_x', 'gyros_arm_y', 'gyros_arm_z', 'accel_arm_x', 'accel_arm_y', 'accel_arm_z', 'magnet_arm_x', 'magnet_arm_y', 'magnet_arm_z', 'kurtosis_roll_arm', 'kurtosis_picth_arm', 'kurtosis_yaw_arm', 'skewness_roll_arm', 'skewness_pitch_arm', 'skewness_yaw_arm', 'max_roll_arm', 'max_picth_arm', 'max_yaw_arm', 'min_roll_arm', 'min_pitch_arm', 'min_yaw_arm', 'amplitude_roll_arm', 'amplitude_pitch_arm', 'amplitude_yaw_arm', 'roll_dumbbell', 'pitch_dumbbell', 'yaw_dumbbell', 'kurtosis_roll_dumbbell', 'kurtosis_picth_dumbbell', 'kurtosis_yaw_dumbbell', 'skewness_roll_dumbbell', 'skewness_pitch_dumbbell', 'skewness_yaw_dumbbell', 'max_roll_dumbbell', 'max_picth_dumbbell', 'max_yaw_dumbbell', 'min_roll_dumbbell', 'min_pitch_dumbbell', 'min_yaw_dumbbell', 'amplitude_roll_dumbbell', 'amplitude_pitch_dumbbell', 'amplitude_yaw_dumbbell', 'total_accel_dumbbell', 'var_accel_dumbbell', 'avg_roll_dumbbell', 'stddev_roll_dumbbell', 'var_roll_dumbbell', 'avg_pitch_dumbbell', 'stddev_pitch_dumbbell', 'var_pitch_dumbbell', 'avg_yaw_dumbbell', 'stddev_yaw_dumbbell', 'var_yaw_dumbbell', 'gyros_dumbbell_x', 'gyros_dumbbell_y', 'gyros_dumbbell_z', 'accel_dumbbell_x', 'accel_dumbbell_y', 'accel_dumbbell_z', 'magnet_dumbbell_x', 'magnet_dumbbell_y', 'magnet_dumbbell_z', 'kurtosis_roll_forearm', 'kurtosis_picth_forearm', 'kurtosis_yaw_forearm', 'skewness_roll_forearm', 'skewness_pitch_forearm', 'skewness_yaw_forearm', 'max_roll_forearm', 'max_picth_forearm', 'max_yaw_forearm', 'min_roll_forearm', 'min_pitch_forearm', 'min_yaw_forearm', 'amplitude_roll_forearm','amplitude_pitch_forearm','amplitude_yaw_forearm','total_accel_forearm','var_accel_forearm','avg_roll_forearm','stddev_roll_forearm','var_roll_forearm','avg_pitch_forearm','stddev_pitch_forearm','var_pitch_forearm','avg_yaw_forearm','stddev_yaw_forearm','var_yaw_forearm','gyros_forearm_x','gyros_forearm_y','gyros_forearm_z','accel_forearm_x','accel_forearm_y','accel_forearm_z','magnet_forearm_x','magnet_forearm_y','magnet_forearm_z']

# cmap = cm.get_cmap('gnuplot')
# scatter = scatter_matrix(x, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(50,50), cmap = cmap)


# In[192]:


# Check for need for scaling
pd.set_option("display.max_rows", None, "display.max_columns", None)
x.describe()


# In[197]:


# Transform scales to 0-1
xcols = x.columns.values.tolist()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_fe_s = scaler.fit_transform(x)
x_fe = pd.DataFrame(x_fe_s, columns = xcols)
x_fe.describe()

# could try standard scaler or other methods to compare and improve
# or
# Scale
# ss = StandardScaler()
# X_train_scaled = ss.fit_transform(X_train)
# X_test_scaled = ss.transform(X_test)


# ------------------------ END FEATURE ENGINEERING ------------------------

# In[ ]:


------------------------ START MODEL TRAINING ------------------------


# In[198]:


# train test split and standardize / scale
X_train, X_test, y_train, y_test = train_test_split(x_fe, y, test_size=0.25, random_state=0)


# In[210]:


# DECISION TREE
from sklearn.tree import DecisionTreeClassifier
start = datetime.now()
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
complete = datetime.now()
time = complete - start
print(" ")
print("Time to Run DecisionTreeClassifier")
print(time)


# In[220]:


# Which features matter the most in determining the classe?
for importance, name in sorted(zip(clf.feature_importances_, X_train.columns),reverse=True)[:10]:
    print (name, importance)


# In[225]:


# Check all accuracy measures
# Build confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

YPred = clf.predict(X_test) 
accuracy = accuracy_score(y_test, YPred)
report = classification_report(YPred, y_test)
cm = confusion_matrix(y_test, YPred)

print("Classification report:")
print("Accuracy: ", accuracy)
print(report)
print("Confusion matrix:")
print(cm)


# In[212]:


# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
start = datetime.now()
rf = RandomForestClassifier().fit(X_train, y_train)
print('Accuracy of Random Forest classifier on training set: {:.2f}'
     .format(rf.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'
     .format(rf.score(X_test, y_test)))
complete = datetime.now()
time = complete - start
print(" ")
print("Time to Run DecisionTreeClassifier")
print(time)


# In[213]:


# XGBOOST
import xgboost as xgb
start = datetime.now()
xgb = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
xgb.fit(X_train, y_train)
print('Accuracy of XGBoost classifier on training set: {:.2f}'
     .format(xgb.score(X_train, y_train)))
print('Accuracy of XGBoost classifier on test set: {:.2f}'
     .format(xgb.score(X_test, y_test)))
complete = datetime.now()
time = complete - start
print(" ")
print("Time to Run DecisionTreeClassifier")
print(time)


# In[214]:


# K-NEAREST NEIGHBORS
from sklearn.neighbors import KNeighborsClassifier
start = datetime.now()
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
complete = datetime.now()
time = complete - start
print(" ")
print("Time to Run DecisionTreeClassifier")
print(time)


# In[215]:


# LINEAR DISCRIMINANT
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
start = datetime.now()
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))
complete = datetime.now()
time = complete - start
print(" ")
print("Time to Run DecisionTreeClassifier")
print(time)


# In[216]:


# GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
start = datetime.now()
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))
complete = datetime.now()
time = complete - start
print(" ")
print("Time to Run DecisionTreeClassifier")
print(time)


# In[217]:


# SUPPORT VECTOR
from sklearn.svm import SVC
start = datetime.now()
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))
complete = datetime.now()
time = complete - start
print(" ")
print("Time to Run DecisionTreeClassifier")
print(time)


# In[218]:


# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
start = datetime.now()
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
complete = datetime.now()
time = complete - start
print(" ")
print("Time to Run DecisionTreeClassifier")
print(time)


# In[ ]:


------------------------ END MODEL TRAINING ------------------------


# In[ ]:


------------------------ START MODEL TUNING ------------------------


# In[ ]:


# Further data exploration, possible model improvement
# sensitivity and specificity - determine best model fit based on these, compare all models
# principle component analysis - could improve or reduce prediction accuracy
# feature importance - to understand model better
# density plots
# confusion matrix
# error rate against number of trees
# random error rate
# cross validation - protect against over fitting
