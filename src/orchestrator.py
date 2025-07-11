import time
from pathlib import Path

import pandas as pd

from top_level_classify import add_case_classification_df
from extract_modules import add_case_modules_df, deduplicate_case_module_names
from cluster_case_summaries import add_case_summary_df, cluster_case_summaries, extract_cluster_topics, deduplicate_clusters

# ROSS
path = "/workspace/aptean/ROSS_case_list.xlsx"

base_debug_path = Path("/workspace/aptean/scratchpad_data")

df = pd.read_excel(path)
df = df.iloc[:100]


# classify all clases (lv 1)
start = time.time()
case_classification_df = add_case_classification_df(df)

#DEBUG save file
case_classification_df_path = base_debug_path.joinpath('case_classification.xlsx')
case_classification_df.to_excel(case_classification_df_path)
print('finished case classification', time.time() - start)



# Get all case modules (lv 2)
start = time.time()
module_names_df = add_case_modules_df(case_classification_df)
deduplicated_module_names_df = deduplicate_case_module_names(module_names_df)

#DEBUG save file
deduplicated_module_name_df_path = base_debug_path.joinpath('deduplicated_module_name.xlsx')
deduplicated_module_names_df.to_excel(deduplicated_module_name_df_path)
print('finished module name extraction',  time.time() - start)



# add case summaries for every case
start = time.time()
case_summary_df = add_case_summary_df(deduplicated_module_names_df)

#DEBUG save file
case_summary_df_path = base_debug_path.joinpath('case_summary.xlsx')
case_summary_df.to_excel(case_summary_df_path)
print('finished case summary addition',  time.time() - start)



# Cluster based on case summary at (category, module) groupings
cluster_df_list = []
base_n_clusters = 10
for cluster_group_col_list, cluster_df in case_summary_df.groupby(['deduplicated_module_name', 'category_name']):
    print(cluster_group_col_list)
    print(len(cluster_df))
    print(f"\n\n{'-'*25}\n\n")
    
#     n_clusters = min(base_n_clusters, len(cluster_df))
#     case_cluster_df = cluster_case_summaries(cluster_df, n_clusters)
#     cluster_topics_df = extract_cluster_topics(case_cluster_df)
#     deduplicated_cluster_df = deduplicate_clusters(cluster_topics_df)

#     cluster_df_list.append(deduplicated_cluster_df)


# src_cols_list = case_summary_df.columns
# extra_cols_list = [col_name for col_name in cluster_df_list[0].columns if col_name not in src_cols_list]

# for col_name in extra_cols_list:
#     case_summary_df[col_name] = None

# for cluster_df in cluster_df_list:
#     cluster_df_index = cluster_df.index

#     for col_name in extra_cols_list:
#         case_summary_df.loc[cluster_df_index, col_name] = cluster_df[col_name]    

# #DEBUG save file
# deduplicated_cluster_df_path = base_debug_path.joinpath('deduplicated_cluster_df.xlsx')
# case_cluster_df.to_excel(deduplicated_cluster_df_path)
# print('finished clustering based on case summary')