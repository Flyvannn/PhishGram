import pandas as pd

g = pd.read_csv('data/grambeddings/gram.csv')
u = pd.read_csv('data/urlData.csv')
u['label'] = u['label'].map(lambda x:0 if x==0 else 1)
df = pd.concat([g,u]).reset_index(drop=True)
df.to_csv('data/all.csv', index=False)

exit(0)
train_df = pd.read_csv('data/grambeddings/train.csv', header=None, encoding='utf-8')
test_df = pd.read_csv('data/grambeddings/test.csv', header=None, encoding='utf-8')

train_df.columns = ['l', 'url', 'n']
test_df.columns = ['l', 'url', 'n']
train_df.drop('n', axis=1, inplace=True) #改变原始数据
test_df.drop('n', axis=1, inplace=True) #改变原始数据

print(train_df.head())
print(test_df.head())

df = pd.concat([train_df, test_df]).reset_index(drop=True)
print(df.l.tolist())
df['label'] = df['l'].map(lambda x: 1 if x == 1 else 0)
print(df.head())

df.drop('l', axis=1, inplace=True) #改变原始数据

print(df.head())

df.to_csv('data/grambeddings/gram.csv', index=False)


