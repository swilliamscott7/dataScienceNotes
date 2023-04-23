# DataViz probably works better in a jupyter notebook # 
# %matplotlib inline ---- for jupyter 

import matplotlib.pyplot as plt 
import seaborn as sns

# nba[nba["fran_id"] == "Knicks"].groupby("year_id")["pts"].sum().plot() # line plot
# nba["fran_id"].value_counts().head(10).plot(kind="bar") # bar chart

# img=plt.boxplot(iris['sepal length (cm)'])
# img=plt.scatter(iris['sepal width (cm)'],iris['sepal length (cm)'])
# plt.show(img)

# change figsize
from matplotlib.pyplot import figure 
_ = figure(figsize(10,10))


plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


_ = df.population.plot(kind='hist', rot=70, logx=True, logy=True)  # dummy variable can prevent unnecessary output from being displayed
plt.show()
df.plot.hist()   #histogram for each column


plt.plot(np_year, np_pop, color='orange')
plt.xlabel('Year', color='red')
plt.ylabel('Population', color='red')
plt.title('World Population Over Time')
# plt.yscale('log')
plt.show()

# scatter 
plt.scatter(year, population, s=calorie_intake, alpha=0.5, c='red')    # s for size gives extra dimension / alpha gives opacity: [0,1] 
plt.text()

from matplotlib.pyplot import figure
%matplotlib inline
genre_boxplot = sns.boxplot(data = genre_df.iloc[:,2:])
genre_boxplot = plt.xlabel('Genres')
genre_boxplot = plt.ylabel('Viewing Duration')
genre_boxplot = figure(figsize=(10,10))
plt.show()

# matplotlib boxplot
boxplot = genre_df.iloc[:,2:].boxplot(figsize= (20,15), rot=90, fontsize=12)      #relies on running a cell below, where I have already filtered out initiation columns 
bxplot = boxplot.axes.set_ylim([-100,10000])
plt.show()


# save dataViz
plt.savefig('plot_name.png', bbox_inches='tight') # N.B. plt.savefig() must be called before plt.show() otherwise png file will be blank 
_ = plt.savefig('cluster.png')


labels = labels_df.index
sizes = labels_df.labels
fig1, ax1 = plt.subplots()
fig1.set_figheight(15)
fig1.set_figwidth(15)
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of clusters amongst entire Sampled data')
plt.show()

# Bar Chart #
ax = commits_per_year.plot.bar(legend=False)
ax.set_title('Commits per year on github')
ax.set_xticklabels(years)
ax.set_xlabel('years')
ax.set_ylabel('number of commits')
plt.show()

# Bar 
years = commits_per_year.index.year


#Box Plots
_ = df.boxplot(column='population', by='continent')    # by is the column name we want to test the box plot across i.e. separate box plot by continent
plt.show()
#Scatter graph 
df.plot(kind='scatter', x = 'Initial_cost', y='total_est_fee', rot=70)
df.plot.scatter(x=, y=)

# Line Graph 
_ = df.plot(x='col1', y='col2')
_ = plt.xlabel('avg')
_ = plt.ylabel('year')
plt.show()

# Plotting multiple dataframes on one plot area 
ax = df1.plot()
df2.plot(ax=ax)
# Example2
_ = yearly1.plot(x='year', y='proportion_deaths', label='yearly1', color='b')
yearly2.plot(x='year', y='proportion_deaths', label='yearly2', ax=_, color='r')
_ = plt.xlabel('Year')   # or _.set_xlabel('')
_ = plt.ylabel('Proportion of deaths')
plt.show()

# plotting a logarithmic scale
COLORS =['blue', 'red'......]
TOP_CAP_TITLE ='Top Bitcoin Market Capitalisations'
_ = df.plot.bar(y='market_cap', logy=True, color=COLORS, title=TOP_CAP_TITLE)

# Creating multiple subplots using plt.subplot 
fig, ax = plt.subplots()
nano_plt, micro_plt, biggish_plt = plt.bar([0, 1, 2], values, tick_label=LABELS)
nano_plt.set_facecolor('darkred')
micro_plt.set_facecolor('darkgreen')
biggish_plt.set_facecolor('darkblue')
ax.set_ylabel('Number of coins')
ax.set_title('Classification of coins by market cap')
plt.show()

# There are two optional arguments for plt.subplots(rows,columns)
fig, ax = plt.subplots(1,2) # produces 2 plots side by side 
fig, ax = plt.subplots(2) # 2 subplots on top of each other 
ax.set_title('Trend in Tea Prices')   # title for plot
fig.suptitle('Project on Tea')        # title for whole figure 

# Multiple plots 
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 0].set_title('Axis [0,0]')
axs[0, 1].plot(x, y, 'tab:orange')
axs[0, 1].set_title('Axis [0,1]')
axs[1, 0].plot(x, -y, 'tab:green')
axs[1, 0].set_title('Axis [1,0]')
axs[1, 1].plot(x, -y, 'tab:red')
axs[1, 1].set_title('Axis [1,1]')
# example2
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('fhfffk')
ax1.plot(x,y)

#### EXAMPLE #####
ax = figure(figsize(20,10))
features = churn_coefficients['Feature']
churn_coefficients['positive'] = churn_coefficients['Churn PPT Benefit %'] > 0
churn_coefficients.loc[(churn_coefficients['Churn PPT Benefit %'] > 0) & (churn_coefficients['significance'] == True), 'colour'] = 'pos'
churn_coefficients.loc[((churn_coefficients['Churn PPT Benefit %'] > 0) | (churn_coefficients['Churn PPT Benefit %'] < 0) ) & (churn_coefficients['significance'] == False), 'colour'] = 'insig'
churn_coefficients.loc[(churn_coefficients['Churn PPT Benefit %'] < 0) & (churn_coefficients['significance'] == True), 'colour'] = 'neg'
ax = churn_coefficients[['Feature', 'Churn PPT Benefit %', 'colour']].plot.barh(legend = False, color = [churn_coefficients['colour'].map({'pos': 'g', 'neg': 'r', 'insig':'y'})]) # x = churn_coefficients['Churn PPT Benefit %'], y = churn_coefficients['Feature'],
ax.set_title('Quarterly Churn PPT Benefit')
ax.set_yticklabels(features)
ax.set_xlabel('Churn PPT Benefit %')
plt.savefig('Churn_PPT_Benefit.png', bbox_inches='tight')
plt.show()

##################################
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pdf.plot(kind='barh', stacked=True, ax=ax, color=colours, width=1)
ax.hlines(np.arange(per_page) + 0.5, 0, 1, linewidth=1)
ax.legend(loc=7, bbox_to_anchor=(1.2, 0.5))
ax.set_xlabel('Contribution', fontweight='bold')
ax.set_ylabel('Component', fontweight='bold')

fig.tight_layout()
plt.close()
fig.savefig('filename.png', bbox_inches = 'tight')


sns.pairplot(df, hue = 'target_variable', diag_kind = 'hist') # good for visualising smaller dim datasets 


#### For loop using seaborn to plot boxplots on scale ######

metrics = ['trend_score','trend_score_thresh', 'idx_trend_score', 'idx_trend_score_thresh','trend_sum_score', 'trend_sum_score_thresh', 'top_5_seen',
           'top_10_seen', 'idx_5rank', 'idx_10rank']

fig, ax = plt.subplots(5,2, figsize=(40,40), squeeze = False) # allows you to provide a single integer reference rather than x,y input
axli = ax.flatten() 
for i,v in enumerate(metrics):
    sns.boxplot(ax = axli[i], x="Target", y=v, data=df_1)
    axli[i].set_xlabel('Target')
    axli[i].set_ylabel(v)
plt.show()


#### For loop using seaborn to make histogram on scale #####

# Plot together 
n_metrics = len(metrics)
fig2, ax2 = plt.subplots(5,2, figsize=(30,30))
axli = ax2.flatten() 
for i,v in enumerate(metrics):
    sns.histplot(df_1.loc[df_1['Target']==1, v], label='DG', bins=30, color = 'red', alpha=0.5, ax = axli[i])
    sns.histplot(df_1.loc[df_1['Target']==0,v], label='Non-DG', bins=30, color = 'orange', alpha=0.5, ax = axli[i] )
    axli[i].set_title('Downgraders')
    axli[i].set_title('Non-downgraders')
    axli[i].legend(loc='upper right')
plt.show()


plt.savefig('plot_name.png', bbox_inches='tight') # N.B. plt.savefig() must be called before plt.show() otherwise png file will be blank 



# Multiple plots 
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 0].set_title('Axis [0,0]')
axs[0, 1].plot(x, y, 'tab:orange')
axs[0, 1].set_title('Axis [0,1]')
axs[1, 0].plot(x, -y, 'tab:green')
axs[1, 0].set_title('Axis [1,0]')
axs[1, 1].plot(x, -y, 'tab:red')
axs[1, 1].set_title('Axis [1,1]')
# example2
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('fhfffk')
ax1.plot(x,y)

# SMOOTH HISTOGRAMS ####
sns.kdeplot()


# CONDITIONAL FORMATTING / HEATMAP PANDAS DF : https://towardsdatascience.com/style-pandas-dataframe-like-a-master-6b02bf6468b0 
df.style.background_gradient(cmap='Blues') # believe this does columnwise conditional formatting

######################################################
