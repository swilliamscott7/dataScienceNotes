# 1. Aggregate users' touchpoints into a single row list element (the fact we ordered by interaction means that touchpoint ordering is preserved) - One row represents one user and their path / journey
# May need to add 'Start','Conversion','Non-Conversion' to path to add context and attribute final touchpoint to conversion
# 2. Count number of transitions (or interactions) between states - For all users, count the total number of transitions from one state to the next (for all possible permutations) - this gives us an absolute transition matrix
# This includes the transition from the final touchpoint to the conversion event
# 3. Calculate transition probabilities between all states in our state-space
# Transform the absolute transition matrix into a probabilistic one (i.e. shows probability of a user transitioning from one state to another)
# Given a touchpoint Y, what is the prob of the next touchpoint being X:
# num transitions to touchpoint X from current touchpoint Y / Total number of interactions with touchpoint Y
# 4. Calculate the removal effect for each of our touchpoints
# Use linear algebra and matrix manipulations

# STEP 1 #
# Find total number of conversions and overall conversion rate
total_conversions = df_paths.conversion.sum()
base_conversion_rate = total_conversions/df_paths.shape[0] # will be 1 if just using conversions dataset
print('Base conversion rate is {:.2f}%'.format(base_conversion_rate*100))

# STEP 2 #
def transition_states(paths):
    
    # Isolate unique channels and all possible transitions
    unique_channels = set(channel for path in paths for channel in path)
    transition_states = {x + '>' + y: 0 for x in unique_channels for y in unique_channels}

    # For each possible state that can be transitioned from, count how many times it transitions to each other possible state
    for possible_state in list(set(unique_channels).difference({'Conversion', 'Null'})):
        for path in paths:
            if possible_state in path:
                state_indices = [index for index, state in enumerate(path) if possible_state == state] # find all indices where each path == possible_state
                for ind in state_indices:
                    transition_states[path[ind] + '>' + path[ind + 1]] += 1
    # ^ I don't like all these nested loops, and this could definitely be done via .apply([list comprehension]) methods, but it exectues quickly for now so I've left it
    
    # Reformat results as a DataFrame
    transition_states = pd.DataFrame(data={'Count':transition_states.values()}, index=transition_states.keys())
    transition_states.index.name = 'Transition'
    
    return transition_states

trans_states = transition_states(df_paths['path'])


## STEP 3 ###
def transition_probas(trans_states):
    
    # Isolate state that is being transitioned from for each transition
    trans_states['State from'] = trans_states.reset_index()['Transition'].apply(lambda x: x.split('>')[0]).values
    
    # Find the cumulative number of transitions made from each possible starting state
    cumulative_counts = trans_states.groupby('State from').sum().rename(columns={'Count':'CumCount'})
    
    # Calculate all transition probabilities
    trans_states = trans_states.reset_index().set_index('State from').join(cumulative_counts).reset_index().set_index('Transition')
    trans_states['Probability'] = trans_states['Count'].divide(trans_states['CumCount'])
    
    # Reformat results as a dictionary
    trans_probas = trans_states.loc[~trans_states['State from'].isin(['Conversion','Null']), 'Probability'].to_dict()

    return trans_probas

trans_probs = transition_probas(trans_states)

### FILL IN TRANSITION MATRIC WITH TRANSITION PROBAS ####
def transition_matrix(paths, transition_probabilities):
    
    """  
    
    Turn our transition probabilities dictionary into a data frame (matrix) 
    Columns represent state they're transitioning to 
    Rows represent state they're transitioning from
    
    """
    
    # Isolate unique channels and initialize a blank transition matrix
    unique_channels = set(x for element in paths for x in element)
    trans_matrix= pd.DataFrame(data=np.zeros((len(unique_channels),len(unique_channels))), columns=unique_channels, index=unique_channels)

    # Set self-transition probabilites for present endpoint states = 1
    for endpoint_state in list(set(['Conversion','Null']).intersection(unique_channels)):
        trans_matrix.at[endpoint_state, endpoint_state] = 1.0

    # Fill trans_matrix in with calculated transition probabilites
    for key, value in transition_probabilities.items():
        origin, destination = key.split('>')
        trans_matrix.at[origin, destination] = value

    return trans_matrix

trans_matrix = transition_matrix(df_paths['path'], trans_probs)

### STEP 4 - REMOVAL EFFECT

# Calculate the impact of removing each channel from our probability model
def removal_effects(trans_matrix, conversion_rate):
    
    """" https://github.com/JesseCastro/markovattribution/tree/master/docs - reference for the maths """
    
    # Initialize removal_effects_dict and isolate removeable channels
    removal_effects_dict = {}
    channels = list(set(trans_matrix.columns).difference(['Start','Null','Conversion']))
    
    # # Adaptation to the original function to make it usable for a dataset containing just conversions # 
    # if 'Null' not in trans_matrix.columns.tolist():
    #     trans_matrix.loc['Null'] = 0 
    #     trans_matrix['Null'] = 0
    #     trans_matrix['Null']['Null'] = 1 # rowsums must equal 1
    
    for channel in channels: # for loop that repeats for each channel that we remove to calculate removal effect of each 
        
        # Create a transition matrix without the possibility of using the given channel # 
        removal_trans_matrix = trans_matrix.copy()
        if 'Null' in trans_matrix.columns:
            removal_trans_matrix['Null'] = removal_trans_matrix['Null'] + removal_trans_matrix[channel]
        else:
            removal_trans_matrix['Null'] = removal_trans_matrix[channel]
        removal_trans_matrix.drop(channel, axis=0, inplace=True)
        removal_trans_matrix.drop(channel, axis=1, inplace=True)
        
        # Matrix R : Columns represent absorption states - i.e. either NULL or Conversion, while rows represent transition states
        absorption_states = removal_trans_matrix[['Null', 'Conversion']].drop(list(set(removal_trans_matrix.index).intersection(['Null', 'Conversion'])), axis=0)
        # Matrix Q : Columns & rows represent transition states (i.e. no absorption states) 
        transition_states = removal_trans_matrix.drop(['Null', 'Conversion'], axis=1).drop(list(set(removal_trans_matrix.index).intersection(['Null', 'Conversion'])), axis=0)

        # Derive the 'Fundamental Matrix' using textbook math : N = inv(It - Q) 
        # The Fundamental matrix has a lot of interesting properties including being able to derive absorprtion probabilities for matrix M
        fundamental_matrix = np.linalg.inv(np.identity(len(transition_states.columns)) - np.asarray(transition_states))
        # Matrix M = N*R (Tells us the absorption probabilities for each absoprtion state, when starting from any transient state) 
        absorption_prob_matrix = np.dot(fundamental_matrix, np.asarray(absorption_states))
        # Only need first row from M because we always start at the start state. Also only need the second column, because we only care about the probability of conversion
        removal_conversion_rate = pd.DataFrame(absorption_prob_matrix, index=absorption_states.index, columns=['Null','Conversion']).loc['Start','Conversion']
        removal_effect = 1 - removal_conversion_rate / conversion_rate
        removal_effects_dict[channel] = removal_effect
    return removal_effects_dict

removal_effects_dict = removal_effects(trans_matrix, base_conversion_rate)

### STEP 5 - MARJOV ATTRIBUTION RESULTS ####

def markov_chain_allocations(removal_effects, total_conversions):
    
    """ """
    # Calculate the summed removal effect of all channels so each can be viewed as a fraction of this
    removal_sum = sum(removal_effects.values())

    return pd.Series({channel: (removal_effect / removal_sum) * total_conversions for channel, removal_effect in removal_effects.items()})

# Volume of conversions attributable to each marketing channel
attributions = markov_chain_allocations(removal_effects_dict, total_conversions)

# Percentage of conversions attributable to each marketing channel 
markov_attributions_manual = np.round((attributions*100/attributions.sum()).sort_values(ascending=False),1)

# Interactions with each marketing channel #
if non_conversions:
    total_channel_interactions = df_conv.append(df_nonconv).channel_new.value_counts()
    total_interactions = total_channel_interactions.sum()
    perc_channel_interactions = total_channel_interactions*100/total_interactions
else:
    total_channel_interactions = df_conv.channel_new.value_counts()
    total_interactions = total_channel_interactions.sum()
    perc_channel_interactions = total_channel_interactions*100/total_interactions

# Relative Markov contributions - we do this for the graphs - normalise by overall presence - to see if outperforming relative to other channels
relative_markov_attributions = np.round((attributions*100/attributions.sum()).sort_values(ascending=False)/perc_channel_interactions
                                        
                                        
### visualise transition matric ##
                                        
fig = plt.figure(figsize=(12,12))
plt.title('Interaction between all states \n', fontsize=24)
sns.heatmap(np.round(trans_matrix*100), cmap='YlGn', annot=True, square=True)
plt.xlabel('State to')
plt.ylabel('State from')
plt.xticks(rotation=65)
plt.show()

                                        

### OR MORE SIMPLY #####
# https://www.channelattribution.net/pdf/ChannelAttributionWhitePaper.pdf
# https://www.channelattribution.net/assets/files/PythonChannelAttribution-c98a8c4eabed0dfd58083870dc807ee0.pdf
# For this, need to remove the start & finish elements from the user path -  # removes start and absorption/end state
!pip install ChannelAttribution
from ChannelAttribution import *
# For Heuristics model to work, need to exclude 'Start' and 'Conversion' from path, else will think these are touchpoints and attribute these to conversion events 
heuristics_results = heuristic_models(paths_channel_only,"path","conversion")

# For Markov Model #
markov_results = markov_model(paths_channel_only,var_path= "path",var_conv="conversion", var_null='non_conversion', out_more=True, order=1)  # ncore = 1, order=1
markov_removal_effects = markov_results['removal_effects']
markov_transition_matrix = markov_results['transition_matrix']
                                        

                                        
