# =============================================================================
# Milestone 6: Analysis codes via Python
# =============================================================================

# run first
import pandas as pd
import numpy as np
import datetime as dt
import string
import time
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation
from math import sqrt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from langdetect import detect
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Part 1:
# Computing covariance and correlation matrix
# =============================================================================

df = pd.read_csv('price_df.csv',index_col = 0)
# find the change in daily closing prices
df['returns'] = df.groupby(['name'])['close'].diff()
print(df)
len(df['name'].unique().tolist())

#transpose data frame according to name, and indexing by date for closing prices only
df_transposed = df.set_index(['name','day']).close.unstack('name')
#compute stock daily return/change
df_return = df_transposed.pct_change()
# create covariance matrix
df_return.cov()
# create correlation matrix
df_return.corr()


# Select stocks with 61 data points to avoid bias correlation
groups = df.groupby(['name'])
groups.get_group('AIRPORT-C8') #only have 1
groups.get_group('3A') # have full records for 3 months
#for more meaningful covariance interpretation, we decided to filter stocks with 61 records ONLY
df_new = groups.filter(lambda x : len(x)==61)
#df_new.to_csv('stocks60.csv')
df_transposed1 = df_new.set_index(['name','day']).close.unstack('name')
df_return1 = df_transposed1.pct_change()
df_return1.cov()
df_return1.corr()



# function defined to compute top positive and negative correlation
#=================================================================================
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_pos_correlations(df, n):
    au_corr = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def get_top_neg_correlations(df, n):
    au_corr = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=True)
    return au_corr[0:n]
#=================================================================================
n = 10

#For top n positive correlated stocks
x = get_top_pos_correlations(df_return, n)
y = get_top_pos_correlations(df_return1, n)

print("Top %d Positive Correlations" % n)
print("For different range of records: ");print(x)
print("Top %d Positive Correlations" % n)
print(" For stocks with >60 records: " );print(y)

#For top n negative correlated stocks
r = get_top_neg_correlations(df_return, n)
s = get_top_neg_correlations(df_return1, n)
    
print("Top %d Negative Correlations" % n)
print("For different range of records: ");print(r)
print("Top %d Negative Correlations" % n)
print(" For stocks with >60 records: " );print(s)


# =============================================================================
# Part 2: Finding stock risks
# =============================================================================

#continued from previous parts. Find stock returns and volatility
dfReturns = df_transposed1.apply(lambda x: np.log(x) - np.log(x.shift(1))).mean()*61
dfReturns = pd.DataFrame(dfReturns)
dfReturns.columns = ['AVGReturns']
dfReturns['Volatility'] = df_transposed1.pct_change().std()*sqrt(61)

#find correlation matrix, i.e. the "similarities" between each stock
corr = df_transposed1.corr()

# generate the linkage matrix for clustering
Z = linkage(corr, 'average')

# function to create dendogram
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

# Hierarchical clustering diagram
max_d = 7  # max_d as in max_distance

fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()

max_d = max_d
clusters = fcluster(Z, max_d, criterion='distance')
cluster_list = clusters.tolist()
dfReturns['clusters'] = cluster_list
from collections import Counter
Counter(clusters)

# plot clusters results with regression line
%matplotlib inline
facet = sns.lmplot(data=dfReturns, x='Volatility', y='AVGReturns', hue='clusters',palette='Set2',fit_reg=True, legend=True, legend_out=True)

#STORE & SAVE SELECTED CLUSTERS
clusterext = dfReturns[dfReturns['clusters']==5]
print(clusterext)
stocknames = clusterext.index.tolist()

#high risk projection
clusterhighext = dfReturns[dfReturns['clusters']==3]
print(clusterhighext)
highstocknames = clusterhighext.index.tolist()

#low risk projection
clusterlowext = dfReturns[dfReturns['clusters']==1]
print(clusterlowext)
lowstocknames = clusterlowext.index.tolist()

#filter newdf based on new stock clusters
df_new2 = pd.DataFrame()
df_new2 = df_new[df_new['name'].isin(stocknames)]
df_new2 = df_new2.dropna()
df_new2.to_csv('stocksmidrisk.csv',index=False)

df_new3 = pd.DataFrame()
df_new3 = df_new[df_new['name'].isin(highstocknames)]
df_new3 = df_new3.dropna()
df_new3.to_csv('stockshighrisk.csv',index=False)

df_new4 = pd.DataFrame()
df_new4 = df_new[df_new['name'].isin(lowstocknames)]
df_new4 = df_new4.dropna()
df_new4.to_csv('stockslowrisk.csv',index=False)


# =============================================================================
# Part 3: Identify valuable stocks - potential stocks for investment
# =============================================================================

# Price to Earnings Ratio = (Market Price of Share) / (Earnings per Share)
# https://www.educba.com/pe-ratio-formula/
# Use financial data crawled earlier
df_fin = pd.read_csv('financial_df.csv')
df_fin.rename(columns={'stockname':'name'}, inplace=True)
df_fin = df_fin[['name','quarter','price','pbt','eps','YoY']]   

#### Need to check which stocks matched the current financial data. 
#### To remove stocks without financial info.
df_low = pd.read_csv('stockslowrisk.csv')
df_low['risk'] = 'low risk'
low_fin = set(df_low['name']).intersection(set(df_fin['name']))
len(low_fin) # 30 companies from low cluster with financial information
low_fin_stocks = list(low_fin)
df_low_fin = pd.merge(df_low,df_fin,how='inner',on='name')
df_low_fin.to_csv('stockslowfin.csv',index=False)

df_mid = pd.read_csv('stocksmidrisk.csv')
df_mid['risk'] = 'mid risk'
mid_fin = set(df_mid['name']).intersection(set(df_fin['name']))
len(mid_fin) #33 companies from mid risk cluster with financial information
mid_fin_stocks = list(mid_fin)
df_mid_fin = pd.merge(df_mid,df_fin,how='inner',on='name')
df_mid_fin.to_csv('stocksmidfin.csv',index=False)

df_hi = pd.read_csv('stockshighrisk.csv')
df_hi['risk'] = 'high risk'
hi_fin = set(df_hi['name']).intersection(set(df_fin['name']))
len(hi_fin) # 80 companies from low cluster with financial information
hi_fin_stocks = list(hi_fin)
df_hi_fin = pd.merge(df_hi,df_fin,how='inner',on='name')
df_hi_fin.to_csv('stockshifin.csv',index=False)

#merge low, mid and high risk stocks that has financial data
Alldf = pd.concat([df_low_fin,df_mid_fin,df_hi_fin], ignore_index=True)
len(Alldf['name'].unique()) #143 companies in total
Alldf.to_csv('combinedstockriskfinancial.csv',index=False)

Alldf = pd.read_csv('combinedstockriskfinancial.csv')
df_fin = Alldf[['name','risk','quarter','price','pbt','eps','YoY']]
df_fin.drop_duplicates(subset=['name','price'],keep='first', inplace=True)

#### Compute P/E ratio, extract PBT & Year-on-Year growth
df_fin['PE_ratio'] = (df_fin['price']/df_fin['eps']) 
df_fin['pbt'] = df_fin.pbt.str.split(',').str.join('').astype(int)
df_fin['pbt_norm'] = df_fin['pbt']
df_fin[['pbt_norm']] = StandardScaler().fit_transform(df_fin[['pbt']])
df_fin['YoY_norm'] = df_fin['YoY'].str.strip('%')
df_fin = df_fin[pd.to_numeric(df_fin['YoY_norm'], errors='coerce').notnull()]
df_fin['YoY_norm'] = ((df_fin['YoY_norm'].astype(float))/100)

df_fin.isnull().values.any()
df_fin.to_csv('allstockfin.csv',index=False)

# Display and explore data 'allstockfin.csv' on Tableau

# potential stocks filter
investstock = df_fin[(df_fin['pbt_norm'] >= 0) & 
                     (df_fin['PE_ratio'] >= 0) &
                     (df_fin['PE_ratio'] <=3) &
                     (df_fin['YoY_norm'] >-1)]
#10 companies from the 33 companies can be considered for investment
invstocknames = investstock['name'].tolist()
len(invstocknames)
print(invstocknames)

# Extract potential gain stocks only for futher analysis i.e. 46 stocks
Alldf['strategy'] = Alldf['name'].apply(lambda x: 'potentialgain' if x in invstocknames else 'potentialloss')
newdf = Alldf[Alldf['strategy']=='potentialgain']

# moving forward, financial data is no longer necessary as well as certain information on stock prices
newdf.drop(['open','quarter','price','pbt','eps','YoY','high','low','strategy'], axis=1, inplace=True)
len(newdf['name'].unique().tolist())
newdf.to_csv('potentialstock.csv',index=False)


# =============================================================================
# Part 4: Sentiment Analysis
# =============================================================================

################
# For Tweets
################
tweetsdf = pd.read_csv('q1tweetstockpot.csv')

tweetsdf['text'] = tweetsdf['text'].astype(str)
tweetsdf['lang'] = tweetsdf['text'].map(lambda x: detect(x)) #takes awhile to process
tweetsdf = tweetsdf[tweetsdf['lang']=='en']

analyzer = SentimentIntensityAnalyzer()
sentiment = tweetsdf['text'].apply(lambda x: analyzer.polarity_scores(x)) 
df4 = pd.concat([tweetsdf,sentiment.apply(pd.Series)],1)
df4.info()
df4.describe()

# Create sentiment polarity classes
df4['summary'] = df4['compound'].apply(lambda x: (x>0 and 'Positive') or (x<0 and 'Negative') or 'Neutral')
df4.to_csv('rawsentitweets.csv',index=False)

df5 = df4.groupby(['name','timestamp'])['compound'].mean().reset_index(name = 'avg_senti')
df5['tweetsenticlass']=df5['avg_senti'].apply(lambda x: (x>0 and 'Positive') or (x<0 and 'Negative') or 'Neutral')
df5.to_csv('finalsentitweets.csv',index=False)

################
# For News
################
dfnews = pd.read_csv('q1newsstockpot.csv')
dfnews['text'] = dfnews['newstitle']+dfnews['newsintro']
analyzer = SentimentIntensityAnalyzer()
sentiment = dfnews['text'].apply(lambda x: analyzer.polarity_scores(x)) 
df6 = pd.concat([dfnews,sentiment.apply(pd.Series)],1)
df6['summary'] = df6['compound'].apply(lambda x: (x>0 and 'Positive') or (x<0 and 'Negative') or 'Neutral')
df6.to_csv('rawsentinews.csv',index=False)

df7 = df6.groupby(['name','dates'])['compound'].mean().reset_index(name = 'avg_senti')
df7['newssenticlass']=df7['avg_senti'].apply(lambda x: (x>0 and 'Positive') or (x<0 and 'Negative') or 'Neutral')
df7.to_csv('finalsentinews.csv',index=False)


# =============================================================================
# Part 5: Merge to create Final Dataset 
# =============================================================================

dfori = pd.read_csv('potentialstock.csv')
dfnewsenti = pd.read_csv('finalsentinews.csv')
dfnewsenti.rename(columns={'dates':'day'}, inplace=True)
dftweetsenti = pd.read_csv('finalsentitweets.csv')
dftweetsenti.rename(columns={'timestamp':'day'}, inplace=True)

merged_df_union = pd.merge(dfori,dfnewsenti,on=['name','day'],how='left')
merged_df_union['avg_senti'].fillna(0, inplace=True)
merged_df_union['newssenticlass'].fillna('none', inplace=True)

merged_df_union2 = pd.merge(merged_df_union,dftweetsenti,on=['name','day'],how='left')
merged_df_union2['avg_senti_y'].fillna(0, inplace=True)
merged_df_union2['tweetsenticlass'].fillna('none', inplace=True)

merged_df_union2.to_csv('mergestocksq1.csv',index=False)




# =============================================================================
# Part 6: Final 4 stocks data selected for Machine Learning classification in SAS
# =============================================================================

# Select 4 stocks with most data points on sentiment scores, from different risk groups
merged_df_union2 = pd.read_csv('mergestocksq1.csv')
stocks4 = ['AAX','MAXIS','NESTLE','AEONCR']
newdf = merged_df_union2[merged_df_union2['name'].isin(stocks4)]

googletrends = pd.read_csv('multiTimeline.csv')
googletrends.reset_index(level=(3,2,1,0), inplace=True)
googletrends.columns = ['day', 'AAX', 'NESTLE', 'MAXIS', 'AEONCR']
googletrends = googletrends.iloc[1:]

googletrends = googletrends.melt(id_vars=['day'])
googletrends.rename(columns={'variable':'name','value':'interest'},inplace=True)
googletrends["day"] = pd.to_datetime(googletrends['day'])
googletrends['day'] = googletrends.day.apply(lambda x: x.strftime('%Y-%m-%d'))


finaldataset = pd.merge(newdf,googletrends,on=['name','day'],how='left')
finaldataset['interest'].fillna('0', inplace=True)
finaldataset.rename(columns={'avg_senti_x':'newspolarity','avg_senti_y':'tweetspolarity'},inplace=True)

# Create target variable on changes of stock returns
finaldataset['target'] = finaldataset['returns'].apply(lambda x: ((x*100)<-1.5 and 'down') or ((x*100)>1.5 and 'up') or 'none')

finaldataset.to_csv('finalstockdata.csv',index=False)

# =============================================================================
# Part 7: Checking the time series i.e. 1D - SAX of the 4 stocks
# =============================================================================

finaldataset = pd.read_csv('finalstockdata.csv')

listnew = finaldataset["name"].unique().tolist()
len(listnew)
df_red = finaldataset.set_index(['name','day']).returns
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
n_paa_segments = 10
n_sax_symbols = 10
n_sax_symbols_avg = 10
n_sax_symbols_slope = 6
for i in listnew:
    records = len(df_red[[i]])
    print("stockname"+str(i))      
    scaleddata = scaler.fit_transform(df_red[[i]])
    #print(scaleddata)      
    paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
    paa_dataset_inv = paa.inverse_transform(paa.fit_transform(scaleddata))
    # SAX transform
    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    sax_dataset_inv = sax.inverse_transform(sax.fit_transform(scaleddata))
    # 1d-SAX transform
    one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                    alphabet_size_slope=n_sax_symbols_slope)
    one_d_sax_dataset_inv = one_d_sax.inverse_transform(one_d_sax.fit_transform(scaleddata))
    plt.figure()
    # First, raw time series
    plt.subplot(1, 2, 1)  
    plt.plot(scaleddata[0].ravel(), "b-")
    plt.title("Raw time series")
    plt.suptitle('Stockname: ' + i,fontsize=16)
    plt.subplot(1, 2, 2)  # Finally, 1d-SAX
    plt.plot(scaleddata[0].ravel(), "b-", alpha=0.4)
    plt.plot(one_d_sax_dataset_inv[0].ravel(), "b-")
    plt.title("1d-SAX, %d symbols (%dx%d)" % (n_sax_symbols_avg * n_sax_symbols_slope,
                                              n_sax_symbols_avg,
                                              n_sax_symbols_slope))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.8, top=0.8)
    plt.show()
