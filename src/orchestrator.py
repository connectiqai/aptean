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
# df = df.iloc[:100]


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

# UNCOMMENT THIS TO RUN THE FINAL PART OF THE PIPELINE ONCE YOU UNDERSTAND WHAT'S HAPPENING AND THE COST IMPLICATIONS

#TODO YOU NEED TO REMOVE THE ILOC STATEMENT TO ACTUALLY RUN THIS FOR ALL THE CASES
#TODO BUG IMPORTANT REMOVE THIS ONLY WHEN YOU EVENTUALLY FIND A MODEL THAT YOU ARE HAPPY WITH IN TERMS OF COST/PERFORMANCE
case_summary_df = case_summary_df.iloc[:100]

groupby_df_list = []
with ThreadPoolExecutor(max_workers=42) as executor:
    groupby_df_cluster_futures2col_list = {}
    for cluster_group_col_list, groupby_df in case_summary_df.groupby(['deduplicated_module_name', 'category_name']):            
        groupby_df_cluster_futures2col_list[executor.submit(add_case_cluster_info, groupby_df)] = cluster_group_col_list
    
    for future in tqdm(as_completed(groupby_df_cluster_futures2col_list), total=len(groupby_df_cluster_futures2col_list)):
        groupby_col_list = groupby_df_cluster_futures2col_list[future]
        groupby_case_cluster_df = future.result()
        
        groupby_df_list.append(groupby_case_cluster_df)


case_clustered_df = pd.concat(groupby_df_list)
case_clustered_df_path = base_debug_path.joinpath('case_clustered.xlsx')
case_clustered_df.to_excel(case_clustered_df_path)
        
