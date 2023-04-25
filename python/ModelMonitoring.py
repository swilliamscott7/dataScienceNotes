# How sensitive is PSI/CSI to n_bins chosen/ quantiles vs fixed bin approach - no perfect solution
# Careful of any imputed values e.g. -999999 /outliers and how these are handled, maybe put in separate bin first
# If fixed bin approach, can be annoiyng if outliers containing 1 or two values each...
# Quantile approach can be problematic where non-unique bin edges i.e. too many samples have same value so cannot split
# granularity of monitoring i.e. compare month / week to train sample. If week then likely more random variability and so CSI > 0.2 


probabilities_dev = lgbm_tuned.predict_proba(X_test_subset)[:,1]
probabilities_oot = lgbm_tuned.predict_proba(X_test_oot_preprocessed.loc[:,[i for i in X_test_subset.columns.tolist()]])[:,1]

def psi_formula(dev_probabilities, oot_probabilities, bins=10):
    """ Calculates PSI for OOT datasets
    
    dev_probabilities : the predicted probabilities on the OOS dataset used at the time of model training 
    oot_probabilities : the predicted probabilities using the same model on the OOT dataset
    
    """

    # Split OOS DEV data into bins and retrieve boundaries, so can apply these to the OOT dataset #
    deciles_oos, bins = pd.qcut(dev_probabilities, bins, retbins=True,labels=False)
    bins[0] = 0
    bins[-1] = 1.0001
    
    # Use these boundary values to assign OOT probability distribution to a decile #
    deciles_oot = pd.cut(oot_probabilities, bins, labels=False, include_lowest=True)
    
    # proportion allocated to each decile in each respective dataset #
    oot_proportions_array = np.array(pd.Series(deciles_oot).value_counts(normalize=True).sort_index())
    oos_proportions_array = np.array(pd.Series(deciles_oos).value_counts(normalize=True).sort_index())
    
    psi_value = np.sum( (oot_proportions_array - oos_proportions_array) * np.log(oot_proportions_array/oos_proportions_array))
    
    return psi_value

#### drift_monitoring.PY - IMPORT THIS MODULE IN ########

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime as dt
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# Is this the right use of decorators ??? 
# When using a staticmethod within a class method, should you use self.staticmethod or ModelMonitoring.staticmethod() - what is the difference??? 

class ModelMonitoring():
    
    def __init__(self,
                 name:str,
                 dev_df:pd.DataFrame,
                 oot_df:pd.DataFrame,
                 continuous_feats:list,
                 categorical_feats:list,
                 quantile_split:bool,
                 num_bins:int,
                 missing_imputation:Union[int, float, complex]=-99999):
        
        """ 
        Constructor
        
            Parameters:
                dev_df: dataset originally used to train (i.e. develop) the model (or at least data from this same time period) 
                oot_df: out of time (relative to dev set) dataset that we want to score using our trained model
        
        """

        self.name = name
        self.dev_df = dev_df
        self.oot_df = oot_df
        self.continuous_feats = continuous_feats
        self.categorical_feats = categorical_feats
        self.quantile_split = quantile_split
        self.num_bins = num_bins
        self.missing_imputation = missing_imputation
    
    @staticmethod
    def get_bins(dev_values:pd.Series,
                 quantile_split:bool,
                 num_bins:int,
                 missing_imputation=-99999) -> list :
        """
        Two approaches here:
            1. Quantiles - i.e. bins created to ensure equal proportion of customers in each bin
            2. Fixed Bins - i.e. equal width bins with differing proportion of customers in each bin
        This now better handles scenarios where an abnormal value (i.e. outside of the current distribution) has been imputed in place of missing value e.g. -99999
        May need some additional flexibility where abnormal values imputed on different scale
        The bins are created on the DEV set and OOT set is allocated according to these (i.e for OOT will allocate to each bin and calculate % of samples that exist within each quantile)
        So if bins=10, DEV will always be [0.1,0.1,0.1.......0.1], but if bins = 5, would be [0.2,0.2,0.2,0.2,0.2], the OOT array in contrast might be [0.1,0.3,0.4,0.1,0.1] suggesting drift...
        """
    
        if quantile_split:
            bins = []
            # missing value imputation e.g. to -99999, can lead to uninformative bins so creates own bin just for these values #
            if dev_values.min() == missing_imputation:
                bins.append((missing_imputation, missing_imputation+1))
                num_bins -= 1
                dev_values = dev_values[dev_values != missing_imputation]
            percents = np.arange(0, num_bins + 1) / (num_bins) * 100
            split_points = [np.percentile(dev_values, p) for p in percents]
            bins.extend((zip(split_points[:-1], split_points[1:])))
        else:
            ''' Fixed size bins'''
            min_val = dev_values.min()
            max_val = dev_values.max()
            
            bins = []
            
            if min_val == missing_imputation:
                bins.append((missing_imputation, missing_imputation+1))
                
                filtered_dev_values = dev_values[dev_values != missing_imputation]
                min_val = filtered_dev_values.min()
                
                num_bins -= 1
            
            split_points = np.linspace(min_val, max_val, num=num_bins+1)
            bins.extend((zip(split_points[:-1], split_points[1:])))

        return bins
    
    @staticmethod
    def calculate_stability_index_value(dev_percentages:list,
                                        oot_percentages:list):
        '''
        Calculate PSI/CSI value from comparing the values
        Replaces percentage values with a very small number if no records in bin (avoids inf results)
        '''
        index_value = 0
        for dev_percentage, oot_percentage in zip(dev_percentages, oot_percentages):
            if dev_percentage == 0:
                dev_percentage = 0.0001
            if oot_percentage == 0:
                oot_percentage = 0.0001
                ## SUM((Actual% – Expected%) * Ln(Actual%/Expected%)) ##
            index_value += (oot_percentage - dev_percentage) * np.log(oot_percentage / dev_percentage)
        if index_value == np.inf:
            index_value = 1
        return index_value
    
    @staticmethod
    def calc_continuous_features(dev_values:pd.Series,
                                 oot_values:pd.Series,
                                 num_bins:int,
                                 quantile_split:bool=True,
                                 missing_imputation=-99999) -> float:
        
        bins = ModelMonitoring.get_bins(dev_values,quantile_split,num_bins,missing_imputation)
        dev_percentages = []
        oot_percentages = []
        for bin_lower_bound, bin_upper_bound in bins:          
            dev_values_count = \
                ((bin_lower_bound <= dev_values) & (dev_values < bin_upper_bound)).sum()
            oot_values_count = \
                ((bin_lower_bound <= oot_values) & (oot_values < bin_upper_bound)).sum()
            
            if (bin_lower_bound, bin_upper_bound) == bins[-1]:
                dev_values_count += (dev_values == bin_upper_bound).sum()
                oot_values_count += (oot_values == bin_upper_bound).sum()
            
            dev_values_percentage = dev_values_count / len(dev_values)
            oot_values_percentage = oot_values_count / len(oot_values)

            dev_percentages.append(dev_values_percentage)
            oot_percentages.append(oot_values_percentage)
        
        index_value = ModelMonitoring.calculate_stability_index_value(dev_percentages, oot_percentages)
        
        return index_value
    
    @staticmethod
    def calc_categorical_features(dev_values:pd.Series,
                                  oot_values: pd.Series) -> float:
        """
        Does not use bins. Every level is treated as a separate bin. We evaluate the proportion of customers within that level in DEV vs OOT.
        If there is a new level but that only contains a small proportion of customers, distribution amongst levels unlikely to change much. Is only when a lot are allocated to the new 
        level will we see a distributional shift triggering a CSI warning
        """
        categories = dev_values.unique()
        dev_percentages = []
        oot_percentages = []
        for category in categories:
            dev_values_count = (dev_values == category).sum()
            oot_values_count = (oot_values == category).sum()
            dev_values_percentage = dev_values_count / len(dev_values)
            oot_values_percentage = oot_values_count / len(oot_values)

            dev_percentages.append(dev_values_percentage)
            oot_percentages.append(oot_values_percentage)

        # Also raise any new/unseen levels # 
        new_levels = [i for i in oot_values.unique() if i not in categories]
        
        index_value = ModelMonitoring.calculate_stability_index_value(dev_percentages, oot_percentages)
        
        return index_value, new_levels

    def drift_all_data(self) -> pd.DataFrame:
    
        """
        Returns a summary dataframe of PSI/CSI scores considering all OOT data vs DEV data i.e. high level view
        SI = Stability Index
        """
        
        cols = self.continuous_feats + self.categorical_feats
        drift_df = pd.DataFrame(columns=['SI-Quantiles','SI-FixedBins'],index=cols) # could add Jensen-Shannon approach
        cat_levels_dict = {}
        for i in cols:
            if i in self.continuous_feats:
                drift_df.loc[drift_df.index==i,'SI-Quantiles'] = ModelMonitoring.calc_continuous_features(dev_values=self.dev_df[i],oot_values=self.oot_df[i],num_bins=self.num_bins,\
                                                                                                          quantile_split=True,missing_imputation=self.missing_imputation)
                drift_df.loc[drift_df.index==i,'SI-FixedBins'] =  ModelMonitoring.calc_continuous_features(dev_values=self.dev_df[i],oot_values=self.oot_df[i],num_bins=self.num_bins,\
                                                                                                           quantile_split=False,missing_imputation=self.missing_imputation)
            if i in self.categorical_feats:
                si_value_cat, new_levels = ModelMonitoring.calc_categorical_features(dev_values=self.dev_df[i],oot_values=self.oot_df[i])
                cat_levels_dict[i] = new_levels
                drift_df.loc[drift_df.index==i,'SI-Quantiles'] = si_value_cat
                drift_df.loc[drift_df.index==i,'SI-FixedBins'] = si_value_cat
        print('----- New levels within categorical features -----\n',cat_levels_dict,'\n----------------------------------------')
        
        return drift_df
    
    def drift_interval_data(self,
                            freq:str='14D',
                            date_col:str='event_dt',
                            performance_metric:str='AUC',
                            target:str='save_flag',
                            visual:bool=True,
                            model_live_ts:pd.Timestamp=pd.Timestamp(2022,11,11)):
        
        """ 
        freq: define the frequency over which to calculate your stability index. Uses standard datetime naming to define period e.g. '1M' would use monthly cadence (see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
        """ 
        # dataframe containing just values within pre-specified time interval #
        timeinterval_df_list = [g for n, g in self.oot_df.groupby(pd.Grouper(key=date_col, freq=freq))]
        date_cadence = [str(n.date()) for n,g in self.oot_df.groupby(pd.Grouper(key=date_col, freq=freq))]
        
        drift_df = pd.DataFrame(columns=[str(n.date()) for n,g in self.oot_df.groupby(pd.Grouper(key=date_col, freq=freq))],index=['count',performance_metric]+self.continuous_feats+self.categorical_feats)
        drift_df.columns = pd.to_datetime(drift_df.columns)
        for i in drift_df.index:
            for index,col_name in enumerate(date_cadence):
                drift_df.loc[drift_df.index=='count',col_name] = timeinterval_df_list[index].shape[0]
                try:
                    drift_df.loc[drift_df.index==performance_metric,col_name] = roc_auc_score(timeinterval_df_list[index].loc[(~timeinterval_df_list[index]['pred_proba'].isna()),target].astype('int64'),\
                                                                                              timeinterval_df_list[index].loc[(~timeinterval_df_list[index]['pred_proba'].isna()),'pred_proba'])
                except ValueError:
                    pass # print('No samples handled by live model in this period')
                if i in self.continuous_feats:
                    drift_df.loc[drift_df.index==i,col_name] = ModelMonitoring.calc_continuous_features(dev_values=self.dev_df[i],oot_values=timeinterval_df_list[index][i],num_bins=self.num_bins,\
                                                                                                        quantile_split=self.quantile_split,missing_imputation=self.missing_imputation)
                if i in self.categorical_feats:
                    drift_df.loc[drift_df.index==i,col_name],_ = ModelMonitoring.calc_categorical_features(dev_values=self.dev_df[i],oot_values=timeinterval_df_list[index][i])
        
        if visual:
            for i in drift_df.index:
                if i in ['count',performance_metric]:
                    _ = plt.figure(figsize=(20,6))
                    _ = sns.lineplot(drift_df.transpose()[i])
                    _ = plt.axvline(x=model_live_ts,color="grey",linestyle=":",label='Model Live',lw=4)
                    if i == 'count':
                        _ = plt.title('Number of eligible LS Calls - {}'.format(self.name))
                    if i == performance_metric:
                        _ = plt.title(performance_metric)
                    plt.legend(loc='upper left',fontsize='large')
                    plt.show()
                else:
                    _ = plt.figure(figsize=(20,6))
                    _ = sns.lineplot(drift_df.transpose()[i])
                    _ = plt.axhline(y=0.1, color="y", linestyle=":",label='SI Warning',lw=4)
                    _ = plt.axhline(y=0.2, color="r", linestyle=":", label='SI Risk',lw=4)
                    _ = plt.axvline(x=model_live_ts,color="grey",linestyle=":",label='Model Live',lw=4)
                    if drift_df.transpose()[i].max()> 0.25:
                        max_val = drift_df.transpose()[i].max()
                    else:
                        max_val = 0.25
                    _ = plt.ylim(bottom=0,top=max_val)
                    _ = plt.title(i)
                    plt.legend(loc='upper left',fontsize='large')
                    plt.show()
                    
        return drift_df
    
def descriptive_views(df:pd.DataFrame,
                      cust_type:str='TV',
                      week_col:str='subs_week_and_year',
                      model_live=dt.date(2022,11,11),
                      event_dt_col='event_dt',
                      segment_col='Level_02_360',
                      target_col='save_flag',
                      target_segment='PROD SCORING',
                      save_path='outputs'):
    """
    High level views of the data that considers:
    a) Target (e.g. save rate) by segment
    b) Target (e.g. save rate) Week on Week
    c) Recent Callers Week on Week
    
    """
    
    # View of Save Rate by segment since Model Live # 
    recent_df = df.copy().loc[df[event_dt_col] >= model_live,]
    segment_df = recent_df.copy().groupby(segment_col).agg({target_col:['mean','count']})
    segment_df.columns = segment_df.columns.get_level_values(1)
    segment_df['prop'] = np.round(100*segment_df['count'] / segment_df['count'].sum(),1)
    segment_df['mean'] = np.round(segment_df['mean'],2)
    segment_df.to_csv('{}/SR_by_segment.csv'.format(save_path))
    display(segment_df)

    # Longitudinal View of Target #
    df_prod = df.copy().loc[df[segment_col] == target_segment,]
    df_prod = df_prod.groupby(week_col).agg({event_dt_col:['min','count'],target_col:['mean']})
    df_prod.columns = df_prod.columns.get_level_values(0)
    df_prod.columns =["event_dt", "callers", "save_flag"]
    df_prod[target_col] = df_prod[target_col]*100
    
    sns.lineplot(data=df_prod, x=event_dt_col,y=target_col)
    plt.title('Save Rate Weekly View')
    plt.xticks(rotation=45)
    plt.xlabel('Week Start Date')
    plt.ylabel('Weekly Save Rate %')
    plt.savefig('{}/WoW_SaveRate.jpg'.format(save_path), dpi = 200, bbox_inches='tight')
    plt.show()
    
    sns.lineplot(data=recent_df.loc[recent_df[segment_col] == target_segment,].groupby(week_col).size())
    plt.title('Weekly Eligible LS Calls Since Launch (including any filters)')
    plt.xlabel('Sky Week')
    plt.savefig('{}/WoW_CallCount.jpg'.format(save_path), dpi = 200, bbox_inches='tight')
    plt.show()

#########
from src.drift_monitoring import * 

mm = ModelMonitoring(cust_type,df_dev,df_oot,continuous_feats=continuous_cols,categorical_feats=categorical_cols,quantile_split=True,num_bins=10,missing_imputation=-99999)
# more holistic view i.e. dev vs sample comparison #
drift_all = mm.drift_all_data()
# More granular perspective #
drift_intervals = mm.drift_interval_data(freq='14D',date_col='event_dt',performance_metric='AUC',target='save_flag',visual=True,mod
