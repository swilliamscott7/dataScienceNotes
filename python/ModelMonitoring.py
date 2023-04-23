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
