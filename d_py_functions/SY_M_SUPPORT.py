import pandas as pd
import numpy as np
import datetime
import random


from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mutual_info_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from scipy.stats import f_oneway, kruskal
import matplotlib.pyplot as plt
import io



def custom_preprocessor(df,
                       data_dictionary_df,
                       index_name):
    
    X = df.copy()
    
    metrics = data_dictionary_df.set_index(index_name)
    all_cols = [c for c in X.columns if c in metrics.index]
    
    
    # Group columns by imputation strategy
    mean_cols   = [c for c in all_cols if metrics.loc[c, 'IMPUTE'] == 'Mean']
    median_cols = [c for c in all_cols if metrics.loc[c, 'IMPUTE'] == 'Median']
    zero_cols   = [c for c in all_cols if metrics.loc[c, 'IMPUTE'] == 'Zero']
    remove_cols = [c for c in all_cols if metrics.loc[c, 'IMPUTE'] == 'Remove']
    
    # Apply scaling logic
    def split_by_scale(cols):
        return {
            'scaled': [c for c in cols if int(metrics.loc[c, 'SCALE']) == 1],
            'unscaled': [c for c in cols if int(metrics.loc[c, 'SCALE']) == 0]
        }

    mean_grp   = split_by_scale(mean_cols)
    median_grp = split_by_scale(median_cols)
    zero_grp   = split_by_scale(zero_cols)
    
    transformers = []

    # Mean impute
    if mean_grp['scaled']:
        transformers.append(('mean_scale', Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler())
        ]), mean_grp['scaled']))
    if mean_grp['unscaled']:
        transformers.append(('mean_only', SimpleImputer(strategy='mean'), mean_grp['unscaled']))

    # Median impute
    if median_grp['scaled']:
        transformers.append(('median_scale', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), median_grp['scaled']))
    if median_grp['unscaled']:
        transformers.append(('median_only', SimpleImputer(strategy='median'), median_grp['unscaled']))

    # Zero impute
    if zero_grp['scaled']:
        transformers.append(('zero_scale', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('scale', StandardScaler())
        ]), zero_grp['scaled']))
    if zero_grp['unscaled']:
        transformers.append(('zero_only', SimpleImputer(strategy='constant', fill_value=0), zero_grp['unscaled']))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # Audit map
    audit = {}
    for name, _, cols in transformers:
        for c in cols:
            audit[c] = name

    return preprocessor, audit, remove_cols

def Heatmap(df,
            correlation=True,
            column_list=[],
            title='Heat Map of Correlation',
            cmap='coolwarm',
            annotate=True,
            x_rotate=0,
            y_rotate=0,
            cbar=True,
            set_center=0,
            figsize=(10,10)):
    
    '''
    Function Which Generates a Heatmap
    
    Parameters:
        Dataframe
        column_name (list): If included, will only show certain columns on the Horizontal Axis.
    
    Returns:
        matlplot plot.
    
    '''
    
    sns.set(style='white')
    
    # View column with Abbreviated title or full. Abbreviated displays nicer.
    if correlation:
        corr = df.corr()
    else:
        corr = df.copy()
    
    if len(column_list)!=0:
        corr = corr[column_list]
    
    mask= np.zeros_like(corr,dtype=bool)
    mask[np.triu_indices_from(mask)]=True
    f,ax = plt.subplots(figsize=figsize)
    
    if len(str(set_center))!=0:
        sns.heatmap(corr,mask=mask,cmap=cmap,center=set_center,square=True,linewidths=1,annot=annotate,cbar=cbar)
    else:
        sns.heatmap(corr,mask=mask,cmap=cmap,square=True,linewidths=1,annot=annotate,cbar=cbar)
    
    
    plt.title(title)
    if y_rotate !=0:
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
            tick.set_horizontalalignment('right')
    if x_rotate !=0:
        plt.xticks(rotation=x_rotate,ha='center', va='top')

    plt.show()


def make_kmeans_pipeline(X,
                         data_dictionary_df,
                         index_name, 
                         n_clusters=8):
    
    '''
    KMeans in scikit-learn requires dense arrays. If your preprocessor uses OneHotEncoder, set sparse_output=False (new) or sparse=False (older versions).
    
    
    '''
    
    preprocessor, audit, remove_cols = custom_preprocessor(X,data_dictionary_df,index_name)
    X_filtered = X.drop(columns=remove_cols)
    
    try:
        # Option in scikit-learn 1.4 > 
        pipe = Pipeline([
            ('prep', preprocessor),
            # n_init='auto' Cant be used. Explore VALUE.
            ('kmeans', KMeans(n_clusters=n_clusters,n_init='auto',random_state=42))
        ])
        
        pipe.fit(X_filtered)
        
    except:
        
        pipe = Pipeline([
            ('prep', preprocessor),
            # n_init='auto' Cant be used. Explore VALUE.
            ('kmeans', KMeans(n_clusters=n_clusters,n_init=10,random_state=42))
        ])
        
        pipe.fit(X_filtered)

    # Runs Pipeline creates new Dataset
    Xt = pipe.named_steps['prep'].transform(X_filtered)  # run prep once
    
    
    try:
        import scipy.sparse as sp
        if sp.issparse(Xt):
            Xt = Xt.toarray()
    except Exception:
        pass    

    # Using Processed Pipeline Return Information as desired
    labels = pipe.named_steps['kmeans'].labels_
    D = pipe.named_steps['kmeans'].transform(Xt)
    
    # Assigned cluster and nearest distance   
    dist_nearest = D[np.arange(D.shape[0]), labels]

    # Next nearest cluster via inf-trick
    D2 = D.copy()
    D2[np.arange(D2.shape[0]), labels] = np.inf
    next_cluster = np.argmin(D2, axis=1)
    dist_next = D2[np.arange(D2.shape[0]), next_cluster]

    # Margin (how much farther the second closest centroid is)
    margin = dist_next - dist_nearest

    # Build output table without contaminating X_filtered used for transforms
    out = X_filtered.copy()
    out['LABEL'] = labels
    out['DISTANCE_PRIM_CENTROID'] = dist_nearest
    out['NEXT_CLUSTER'] = next_cluster
    out['DISTANCE_SECOND_CENTROID'] = dist_next
    out['MARGIN_DISTANCE'] = margin
    
    
    scaled = pd.DataFrame(Xt,columns=X_filtered.columns)
    
    return pipe, audit, out, Xt, scaled


def create_clustering_visualization(df,pipeline_object,name_map={}):
    
    
    df = df.copy()  # preprocessed dense matrix
    labels = pipeline_object.named_steps['kmeans'].labels_
    centers = pipeline_object.named_steps['kmeans'].cluster_centers_
    
    if len(name_map)==0:
        name_map = {x:f"Cluster {x}" for x in range(len(centers))}
    
    pca = PCA(n_components=2, random_state=42)
    df_2d = pca.fit_transform(df)
    centers_2d = pca.transform(centers)

    # Plot
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(df_2d[:,0], df_2d[:,1], c=labels, s=15, cmap='tab10', alpha=0.7)
    plt.scatter(centers_2d[:,0], centers_2d[:,1], c=range(centers_2d.shape[0]),
                cmap='tab10', s=100, marker='o', edgecolor='black', linewidth=1.5)
    plt.title("Mathematical Representation of Centroid Clusters and Distribution in 2D")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])

    # Update legend
    
    handles, labels_legend = scatter.legend_elements()

    # Clean labels: remove LaTeX formatting and convert to int
    clean_labels = [int(lbl.strip('$\\mathdefault{}')) for lbl in labels_legend]

    # Map to business names
    business_labels = [name_map.get(lbl, str(lbl)) for lbl in clean_labels]
    
    plt.legend(handles, business_labels, title="Cluster Legend", loc="best", frameon=True)
    plt.grid(alpha=0.25)
    plt.show()


def scatter_from_dataframe(df, 
                           x_col, 
                           y_col, 
                           x_label=None,
                           y_label=None,
                           color_col=None,
                           title="Scatter Plot",
                           x_axis_limit=[0],
                           y_axis_limit=[0]):
    """
    Creates a scatter plot from a DataFrame with optional color mapping and axis limits.
    action: "show" (display), "save" (save to file), "memory" (return BytesIO object)
    save_path: required if action="save"
    """
    plt.figure(figsize=(10, 5))

    if color_col:
        scatter = plt.scatter(df[x_col], df[y_col], c=df[color_col], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label=color_col)
    else:
        plt.scatter(df[x_col], df[y_col], alpha=0.7)

    # Axis limits
    if len(x_axis_limit) == 2:
        plt.xlim(x_axis_limit[0], x_axis_limit[1])
    elif len(x_axis_limit) == 1:
        plt.xlim(x_axis_limit[0], df[x_col].max() * 1.10)

    if len(y_axis_limit) == 2:
        plt.ylim(y_axis_limit[0], y_axis_limit[1])
    elif len(y_axis_limit) == 1:
        plt.ylim(y_axis_limit[0], df[y_col].max() * 1.10)

    if not x_label:
        x_label = x_col
        
    if not y_label:
        y_label = y_col
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    
    # Tried to save them and show them as a single Image, did not work
    plt.show()


def sample_score_silhouette(df,label,metric='euclidean',sample_size=10000):
    
    '''
    Close to +1: Clusters are well-separated and points are near their own cluster center.
    Around 0: Clusters overlap or points are near the decision boundary.
    Negative: Many points are likely assigned to the wrong cluster.
    
    
    '''
    
    import time
    
    start_time = time.perf_counter()
    temp_score = silhouette_score(df, 
                                  label, 
                                  metric=metric,
                                  sample_size=sample_size)

    end = time.perf_counter() - start_time
    print(f"Run Time: {end:.4f}, Silhouette Score: {temp_score:.4f}")
    return {'time':end,'score':temp_score}


def iteratively_test_sampled_model_performance(model_testing_function,
                                               sample_size=10000,
                                               iterations_=10,
                                               maximum_run_time=100,
                                               threshold=.005,
                                               **kwargs):

    '''
    Make Comment abbout NEEDING SAMPLE_SIZE to WORK. CLear Purpose of this, can make models for other types of Performance
    Make Comment about required input
    
    '''

    std_dev = 1
    run_time = 0
    time_out=0
    
    while  std_dev>threshold:
        score_list = []
    
        for iteration in range(iterations_):
            score_list.append(model_testing_function(sample_size=sample_size,**kwargs))
        
        mean_ = np.mean([x['score'] for x in score_list])
        std_dev = np.std([x['score'] for x in score_list],ddof=1)
        run_time += np.sum([x['time'] for x in score_list])
        
        print(f'\nMean Silhouette Score: {mean_:.4f}, Standard Deviaiton: {std_dev:.4f}, Run Time: {run_time:.2f}')    
        
        if run_time > maximum_run_time:
            print("Operation Timed Out, over 100 Seconds of Compute")
            return {'score':mean_,'std_dev':std_dev,'run_time':run_time,'timed_out':1}
                    
        if std_dev > threshold:
            print('Standard Deviation is Greater than Threshold, Increase Sample Size by 25% and Try Again.')
            sample_size = int(sample_size*1.25)
            
    return {'score':mean_,'std_dev':std_dev,'run_time':run_time,'timed_out':time_out}


def feature_importance_by_clusters(df, cluster_labels=None, features=None):
    """
    Compute feature importance for clustering:
    - ANOVA F-statistic and p-value
    - Kruskal-Wallis H-statistic and p-value
    - Eta-squared and Omega-squared effect sizes
    """
    if cluster_labels is None:
        if 'LABEL' not in df.columns:
            raise ValueError("Provide cluster_labels or include 'LABEL' in df.")
        cluster_labels = df['LABEL'].values
    else:
        cluster_labels = np.asarray(cluster_labels)

    if len(cluster_labels) != len(df):
        raise ValueError("cluster_labels length must equal number of rows in df.")

    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f != 'LABEL']

    unique_labels = np.unique(cluster_labels)
    k = len(unique_labels)
    n = len(df)

    results = []
    for f in features:
        x = df[f].astype(float)
        groups = [x[cluster_labels == c].dropna().values for c in unique_labels]

        valid_groups = [g for g in groups if len(g) >= 2]
        if len(valid_groups) < 2:
            results.append([f, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        # ANOVA
        try:
            F, pA = f_oneway(*valid_groups)
        except:
            F, pA = np.nan, np.nan

        # Kruskal
        try:
            H, pK = kruskal(*valid_groups)
        except:
            H, pK = np.nan, np.nan

        # Effect sizes
        global_mean = x.mean()
        SS_between = sum(len(g) * (g.mean() - global_mean) ** 2 for g in valid_groups)
        SS_within = sum(((g - g.mean()) ** 2).sum() for g in valid_groups)
        SS_total = SS_between + SS_within
        eta_sq = SS_between / SS_total if SS_total > 0 else np.nan
        MS_within = SS_within / max(n - k, 1)
        omega_sq = ((SS_between - (k - 1) * MS_within) /
                    (SS_total + MS_within)) if (SS_total + MS_within) > 0 else np.nan

        results.append([f, F, pA, H, pK, eta_sq, omega_sq])

    out = pd.DataFrame(results, columns=[
        'Feature', 'ANOVA_F', 'ANOVA_p', 'Kruskal_H', 'Kruskal_p', 'Eta_sq', 'Omega_sq'
    ]).sort_values(['Omega_sq', 'Eta_sq', 'ANOVA_F'], ascending=False).reset_index(drop=True)

    return out

def mutual_information_per_feature(df, cluster_labels, features=None, n_bins=12):
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f != 'LABEL']

    cluster_labels = np.asarray(cluster_labels)
    results = []
    for f in features:
        x = df[f].values
        mask = ~np.isnan(x)
        x = x[mask]
        y = cluster_labels[mask]
        if np.nanstd(x) == 0:
            mi = 0.0
        else:
            edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
            if len(edges) < 3:
                mi = 0.0
            else:
                xb = np.digitize(x, edges[1:-1], right=False)
                mi = mutual_info_score(xb, y)
        results.append([f, mi])
    return pd.DataFrame(results, columns=['Feature', 'Mutual_Info']).sort_values('Mutual_Info', ascending=False)

    
    
