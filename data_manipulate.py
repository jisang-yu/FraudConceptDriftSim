import pandas as pd

df = pd.read_csv(r'./uscecchini.csv', index_col=False)
features = ['fyear', 'gvkey', 'misstate', 'at', 'cogs', 'lt', 'ni', 'ppegt']
high = .99
low = .01
for x in features[3:]:
    df.loc[(df[x]>df[x].quantile(high, interpolation='nearest')), x]=df[x].quantile(high, interpolation='nearest')
for x in features[3:]:
    df.loc[(df[x]<df[x].quantile(low, interpolation='nearest')), x] = df[x].quantile(low, interpolation='nearest')
print(df[features].iloc[:,2:].quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]))

df.to_csv(r'./uscecchini_manip.csv', index_label=False)