import networkx as nx
import pandas as pd
import numpy as np

def calculate_relationship_edge_score(df,
                                      primary_id='ACCTNBR',
                                      secondary_id='PERSNBR',
                                      weight='STRENGTH'):
    
    a = df.rename(columns={secondary_id:'p1',weight:'s1'})
    b = df.rename(columns={secondary_id:'p2',weight:'s2'})

    pairs = (
        a.merge(b,on=primary_id,how='inner')
        .query('p1<p2')
        .assign(contrib=lambda d:d[['s1','s2']].min(axis=1))
    )

    pair_scores = (
        pairs.groupby(['p1','p2'],as_index=False)
        .agg(shared_account=(primary_id,'nunique'),
            shared_strength=('contrib','sum'))
    )

    footprint = (
        strength_df.groupby(secondary_id,as_index=False)
        .agg(footprint=(weight,'sum'),
             n_accounts=(primary_id,'nunique'))
            )

    pair_scores = pair_scores.merge(
        footprint.rename(columns={secondary_id:'p1','footprint':'fp1','n_accounts':'n_accounts_p1'}),
        on='p1',how='left'
    ).merge(footprint.rename(columns={secondary_id:'p2','footprint':'fp2','n_accounts':'n_accounts_p2'}),
        on='p2',how='left'
    )

    pair_scores['rel_score'] = pair_scores['shared_strength']/np.sqrt(pair_scores['fp1']*pair_scores['fp2'])
    pair_scores['rel_score'] = pair_scores['rel_score'].fillna(0)
    
    return pair_scores



def calculate_relationship_score(pair_score_df,
                                 entity_1='p1',
                                 entity_2='p2',
                                 relationship_score='rel_score',
                                 group_id_name= 'Relationship_ID',
                                 relationship_threshold=.35):

    edges_use = pair_scores[pair_scores['rel_score']>=relationship_threshold]

    G = nx.Graph()
    G.add_weighted_edges_from(
    edges_use[[entity_1,entity_2,relationship_score]].itertuples(index=False,name=None),
    weight=relationship_score
    )

    components = list(nx.connected_components(G))

    person_to_group = {}
    for i, comp in enumerate(components,start=1):
        for person in comp:
            person_to_group[person]= i

    group_df = pd.DataFrame({secondary_id:list(person_to_group.keys()),
                             group_id_name:list(person_to_group.values())})


    # Include Standalone Relationships
    all_people = strength_df[secondary_id].unique()
    missing = [p for p in all_people if p not in person_to_group]

    if missing:
        start = group_df['group_id'].max() if len(group_df) else 0
        add = pd.DataFrame({secondary_id:missing, 'group_id':range(start+1,start+1+len(missing))})
        group_df = pd.concat([group_df,add],ignore_index=True)

    return group_df

