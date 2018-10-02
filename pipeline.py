
# coding: utf-8

# In[53]:


import pandas as pd
import ml_metrics as metrics


# In[54]:

print("reading in csv files...")
destinations = pd.read_csv("destinations.csv")
test = pd.read_csv("small_test.csv")
train = pd.read_csv("small_train.csv")


# In[55]:

print( "the shape of training data is: " + str(train.shape))


print("the values of the hotel cluster column: " + str(train["hotel_cluster"].value_counts()))


# In[60]:


#Converting the date_time object in train.csv to a datetime value


# In[61]:


train["date_time"] = pd.to_datetime(train["date_time"])
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month


# In[62]:


import random

unique_users = train.user_id.unique()

sel_user_id = random.sample(list(unique_users),10)
sel_train = train[train.user_id.isin(sel_user_id)]


#splitting our random sample into training and test sets
#everything before July 2014 is in t1 and after goes to t2
t1 = train
t2 = test


# In[66]:


#removing ckick events. test only contains booking events so we need to change t2 
#accordingly by setting is_booking to true for all in t2
#t2 = t2[t2.is_booking == True]


# In[67]:


# Let's create our baseline by always predicting the most common hotel clusters
most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
predictions = [most_common_clusters for i in range(t2.shape[0])]


# In[73]:


# We see 150 rows of anonymized data about every destination. These are probably details like name, location, num of pools
# and other characteristics. TO reduce runtime let us compress these columns using PCA. We'll compress columns d1-d149 into 
# three columns


# In[74]:


from sklearn.decomposition import PCA

pca = PCA(n_components=3)
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small)
dest_small["srch_destination_id"] = destinations["srch_destination_id"]


# ## Lets generate some features
#     Generate new date features based on date_time, srch_ci, and srch_co.
#     
#     Remove non-numeric columns like date_time.
#     
#     Add in features from dest_small.
#     
#     Replace any missing values with -1.

# In[76]:


def calc_fast_features(df):
	dfc = df.copy()
	df.loc[:,"date_time"] = pd.to_datetime(dfc["date_time"])
	df.loc[: ,"srch_ci"] = pd.to_datetime(dfc["srch_ci"], format='%Y-%m-%d', errors="coerce")
	df.loc[: ,"srch_co"] = pd.to_datetime(dfc["srch_co"], format='%Y-%m-%d', errors="coerce")
	
	props = {}
	for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
		props[prop] = getattr(df["date_time"].dt, prop)
	
	carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
	for prop in carryover:
		props[prop] = df[prop]
	
	date_props = ["month", "day", "dayofweek", "quarter"]
	for prop in date_props:
		props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
		props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
	props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')
		
	ret = pd.DataFrame(props)
	
	ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
	ret = ret.drop("srch_destination_iddest", axis=1)
	return ret

df = calc_fast_features(t1)
df.fillna(-1, inplace=True)


# ## Finally introducing machine learning.
# 
# We'll use a Random Forest Classifier as our model. Random forests build trees which can allow us 
# to predict non linear tendencies in the data

# In[ ]:


predictors = [c for c in df.columns if c not in ["hotel_cluster"]]
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
scores = cross_validation.cross_val_score(clf, df[predictors], df['hotel_cluster'], cv=3)


# That really isn't a good accuracy. This is to be expected when training on 100 clusters/labels. Instead we can try and use 100 binary classifiers. Each classifier will determine if a row is in it's cluster or not. Kind of like one-hot encoding but with classifiers and labels

# We'll again train Random Forests, but each forest will predict only a single hotel cluster. We'll use 2 fold cross validation for speed, and only train 10 trees per label.
# 
# In the code below, we:
# 
#     -Loop across each unique hotel_cluster.
#     -Train a Random Forest classifier using 2-fold cross validation.
#     -Extract the probabilities from the classifier that the row is in the unique hotel_cluster
#     -Combine all the probabilities.
#     -For each row, find the 5 largest probabilities, and assign those hotel_cluster values as predictions.
#     -Compute accuracy using mapk.
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from itertools import chain

all_probs = []
unique_clusters = df["hotel_cluster"].unique()
for cluster in unique_clusters:
	df["target"] = 1
	df["target"][df["hotel_cluster"] != cluster] = 0
	predictors = [col for col in df if col not in ['hotel_cluster', "target"]]
	probs = []
	cv = KFold(len(df["target"]), n_folds=2)
	clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
	for i, (tr, te) in enumerate(cv):
		clf.fit(df[predictors].iloc[tr], df["target"].iloc[tr])
		preds = clf.predict_proba(df[predictors].iloc[te])
		probs.append([p[0] for p in preds])
	full_probs = chain.from_iterable(probs)
	all_probs.append(list(full_probs))

prediction_frame = pd.DataFrame(all_probs).T
prediction_frame.columns = unique_clusters
def find_top_5(row):
	return list(row.nlargest(5).index)

preds = []
for index, row in prediction_frame.iterrows():
	preds.append(find_top_5(row))


# ## Top clusters based on hotel_cluster
# There are a few Kaggle Kernels for the competition that involve aggregating hotel_cluster based on orig_destination_distance, or srch_destination_id. Aggregating on orig_destination_distance will exploit a data leak in the competition, and attempt to match the same user together. Aggregating on srch_destination_id will find the most popular hotel clusters for each destination. We'll then be able to predict that a user who searches for a destination is going to one of the most popular hotel clusters for that destination.
# 
# We can first generate scores for each hotel_cluster in each srch_destination_id. We'll weight bookings higher than clicks. This is because the test data is all booking data, and this is what we want to predict. We want to include click information, but downweight it to reflect this. Step by step, we'll:
# 
#     -Group t1 by srch_destination_id, and hotel_cluster.
#     -Iterate through each group, and:
#         -Assign 1 point to each hotel cluster where is_booking is True.
#         -Assign .15 points to each hotel cluster where is_booking is False.
#         -Assign the score to the srch_destination_id / hotel_cluster combination in a dictionary.
# Here's the code to accomplish the above steps:

# In[ ]:


def make_key(items):
	return "_".join([str(i) for i in items])

match_cols = ["srch_destination_id"]
cluster_cols = match_cols + ['hotel_cluster']
groups = t1.groupby(cluster_cols)
top_clusters = {}
for name, group in groups:
	clicks = len(group.is_booking[group.is_booking == False])
	bookings = len(group.is_booking[group.is_booking == True])
	
	score = bookings + .15 * clicks
	
	clus_name = make_key(name[:len(match_cols)])
	if clus_name not in top_clusters:
		top_clusters[clus_name] = {}
	top_clusters[clus_name][name[-1]] = score


# At the end, we'll have a dictionary where each key is an srch_destination_id. Each value in the dictionary will be another dictionary, containing hotel clusters as keys with scores as values. Here's how it looks:

# '76': {52: 0.15, 66: 0.3, 76: 0.15, 87: 0.8999999999999999, 90: 0.15},  
# '87': {46: 0.6, 58: 0.3, 63: 0.3},

# We'll next want to transform this dictionary to find the top 5 hotel clusters for each srch_destination_id. In order to do this, we'll:
# 
#     -Loop through each key in top_clusters.
#     -Find the top 5 clusters for that key.
#     -Assign the top 5 clusters to a new dictionary, cluster_dict.

# In[ ]:


import operator

cluster_dict = {}
for n in top_clusters:
	tc = top_clusters[n]
	top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
	cluster_dict[n] = top


# Once we know the top clusters for each srch_destination_id, we can quickly make predictions. To make predictions, all we have to do is:
# 
#     -Iterate through each row in t2.
#     -Extract the srch_destination_id for the row.
#     -Find the top clusters for that destination id.
#     -Append the top clusters to preds.

# In[ ]:


preds = []
for index, row in t2.iterrows():
	key = make_key([row[m] for m in match_cols])
	if key in cluster_dict:
		preds.append(cluster_dict[key])
	else:
		preds.append([])


# By the end of the loop preds will be a list of lists containing our cluster predictions

# preds = [...[50, 4, 59, 21, 49],  
#           [25, 4, 50, 42, 48],  
#           [7, 32, 13, 18, 28], ...]

# In[ ]:


#metrics.mapk([[l] for l in t2["hotel_cluster"]], preds, k=5)


# A post in the forums details a data leak that allows you to match users in the training set from the testing set using a set of columns including user_location_country, and user_location_region.
# 
# We'll use the information from the post to match users from the testing set back to the training set, which will boost our score. Based on the forum thread, its okay to do this, and the competition won't be updated as a result of the leak.

# In[ ]:


match_cols = ['user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance']

groups = t1.groupby(match_cols)
	
def generate_exact_matches(row, match_cols):
	index = tuple([row[t] for t in match_cols])
	try:
		group = groups.get_group(index)
	except Exception:
		return []
	clus = list(set(group.hotel_cluster))
	return clus

exact_matches = []
for i in range(t2.shape[0]):
	exact_matches.append(generate_exact_matches(t2.iloc[i], match_cols))


# In[ ]:


def f5(seq, idfun=None): 
	if idfun is None:
		def idfun(x): return x
	seen = {}
	result = []
	for item in seq:
		marker = idfun(item)
		if marker in seen: continue
		seen[marker] = 1
		result.append(item)
	return result
	
full_preds = [f5(exact_matches[p] + preds[p] + most_common_clusters)[:5] for p in range(len(preds))]
#metrics.mapk([[l] for l in t2["hotel_cluster"]], full_preds, k=5)


# In[ ]:


t2.head()


# In[ ]:


write_p = [" ".join([str(l) for l in p]) for p in full_preds]
write_frame = ["{0},{1}".format(t2.index.values[i], write_p[i]) for i in range(len(full_preds))]
write_frame = ["id,hotel_cluster"] + write_frame
with open("predictions.csv", "w+") as f:
	f.write("\n".join(write_frame))

