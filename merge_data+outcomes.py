import tensorflow as tf
import pandas as pd
import os

# Dataset with outcomes
outcomes = "/Users/ajadhav0517/Box/mesa/mesa_nhlbi/Primary/Exam5/Data/mesae5_drepos_20151101.csv"
df = pd.read_csv(outcomes)
df = df[['mesaid', 'htn5c']].dropna()

df2 = pd.read_csv('Data.csv', header=None)
df2.rename(columns={0:'mesaid'}, inplace=True)


mergedDF = pd.merge(df, df2, on='mesaid', how='inner')
print(mergedDF.shape)
print(mergedDF.head())
os.remove('Data+Outcomes.csv')
mergedDF.to_csv('Data+Outcomes.csv', index=False)

