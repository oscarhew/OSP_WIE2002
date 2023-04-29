import pandas as pd

# Read the dataset with no headers
df = pd.read_csv('/Users/simjiahong/Downloads/set4/qrels.trec8.adhoc', delimiter=' ', header=None)

# Drop column 2 (which is column 1 in 1-based indexing)
df = df.drop(columns=[1])

# Rename columns
df.columns = ['topic_id', 'document_id', 'is_relevant']

# Output to a new file
df.to_csv('/Users/simjiahong/Downloads/set4/clean_qrels.trec8.adhoc.csv', index=False)



import pandas as pd

# Read the dataset with no headers and drop columns 2 and 6
df = pd.read_csv('/Users/simjiahong/Downloads/set4/input.ric8tpx', delimiter='\t', header=None).drop(columns=[1, 5])

# Rename columns and select top 10 rows for each topic_id
df.columns = ['topic_id', 'document_id', 'rank', 'similarity_score']
df_top10 = df.groupby('topic_id').head(10)

# Merge with dataset 2 on document_id and topic_id
merged_df = pd.merge(df_top10, pd.read_csv("/Users/simjiahong/Downloads/set4/clean_qrels.trec8.adhoc.csv"), on=["document_id", "topic_id"], how="inner")

# Calculate the probability of relevant and add a new column to the merged dataset
prob_relevant = merged_df.groupby('topic_id')['is_relevant'].sum() / 10
merged_df['prob_relevant'] = pd.Series([None]*len(merged_df))
merged_df.loc[merged_df.groupby('topic_id').tail(1).index, 'prob_relevant'] = prob_relevant.values

# Drop unnecessary columns and rows with null values for prob_relevant
final_df = merged_df.drop(['document_id', 'rank', 'similarity_score', 'is_relevant'], axis=1).dropna(subset=['prob_relevant']).reset_index(drop=True)

# Output to a new file
final_df.to_csv('/Users/simjiahong/Downloads/set4/clean_input.ric8tpx.csv', index=False)


