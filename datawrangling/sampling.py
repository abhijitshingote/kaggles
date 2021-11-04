import numpy as np
import pandas as pd

a=np.random.randint(0,5,[50,5])
df=pd.DataFrame(a,columns=['col1','col2','col3','col4','col5'])
print(df.head())

# Sampling
samples=np.random.choice(df.index,5)
print(df[df.index.isin(samples)])


# X_unselected, X_selected, y_unselected, y_selected = train_test_split(X, y, stratify=y, random_state=123, test_size=10000)

# Lets sample based on col3 - stratified
strats=df.col3.value_counts().index
print(strats)
stratified_indices=[]
for strat in strats:
	indexes=df[df['col3']==strat].index
	indexes=np.random.choice(indexes,3)
	# print(indexes)
	stratified_indices.extend(indexes.tolist())
print(stratified_indices)
print(df[df.index.isin(stratified_indices)].sort_values(['col3']))
