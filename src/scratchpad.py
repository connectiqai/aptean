import pandas as pd
import numpy as np

from pathlib import Path
import shutil
import sklearn.cluster
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import requests
import anthropic
from openai import OpenAI
import voyageai
import voyageai.client
from google import genai
from google.genai import types
import cohere

import sklearn

import time
import json
import os

from pydantic import BaseModel, Field
from PydanticAdaptorAnthropic import PydanticAdaptorAnthropic
from PydanticAdaptorOpenRouter import PydanticAdaptorOpenRouter

from dotenv import load_dotenv

load_dotenv()

# # ROSS
# path = "/workspace/ROSS_case_list.xlsx"

# # Made 2 Manage
# # path = "/Users/suryakrishnan/Documents/GitHub/aptean/excel_experiment/Made2Manage_case_list.xlsx"

# df = pd.read_excel(path)
# # print(len(df))
# # print(df.columns)



case_info_dirpath = "case_info"
case_info_filename = "case_info.md"
model_summary_filename = "model_summary.md"

case_summary_list_filename = "case_summary_list.json"
case_number_filename = "case_number.json"
summary_embedding_filename = "case_summary_embeddings.npy"
cluster_topics_filename = "cluster_topics.md"

case_info_dirpath = Path(case_info_dirpath)
case_summary_list_filepath = case_info_dirpath.joinpath(case_summary_list_filename)
case_number_filepath = case_info_dirpath.joinpath(case_number_filename)
summary_embedding_filepath = case_info_dirpath.joinpath(summary_embedding_filename)
cluster_topics_filepath = case_info_dirpath.joinpath(cluster_topics_filename)

os.makedirs(case_info_dirpath, exist_ok=True)

def make_ollama_request(prompt):
    base_path = "http://localhost:11434/api/generate"
    response = requests.post(
        base_path,
        json={
            "model": "llama3.2:latest",
            "prompt": prompt,
            "stream": False
        }

    )

    model_response = response.json()['response']
    return model_response

def make_openrouter_request(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        # pass api_key here or set it in env
    )

    resp = client.chat.completions.create(
        model="meta-llama/llama-4-maverick",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4000,
    )

    repsonse_text = resp.choices[0].message.content

    return repsonse_text



def get_case_info(row):
    case_info_str = ""

    # key phrases
    col_name_list = ['Title', 'Problem', 'Subject']
    key_phrase_str = ""
    for col_name in col_name_list:
        val = row.loc[col_name]
        if not pd.isna(val):
            key_phrase_str += f"\n- {val}"

    if key_phrase_str:
        key_phrase_str = f"<SUMMARY PHRASES>{key_phrase_str}\n</SUMMARY PHRASES>"
        case_info_str += f"{key_phrase_str}\n\n"


    col_name_tag_list = [('Description', 'DESCRIPTION'), ('Posts', 'POSTS'), ('Resolution Summary', 'RESOLUTION SUMMARY')]
    for (col_name, col_tag) in col_name_tag_list:
        val = row.loc[col_name]
        if not pd.isna(val):
            case_info_str += f"<{col_tag}>\n{val}\n</{col_tag}>\n\n"


    # print(case_info_str)

    return case_info_str

tag_info = """Here is some context on the different pieces of information you may be given
access to as part of analysing the case. Each of the below menitoned tags may or may not be
present for each case

<TAG INFO>
1. SUMMARY PHRASES - These include some phrases used by the customer support analyst incharge of resolving this case to describe the case during various times in their process of tackling this support ticket.
2. DESCRIPTION - This represents the description of the case and some notes around them used by the internal support team in discussing this cases amongst themselves as they work to resolve it.
3. POSTS = This represents really adhoc and unstructured information that were present in the support team's logs as they work to resolve this issues. This can often contain a lot of noise around unneccsary things for you such as scheduling meetings.
4. RESOLUTION SUMMARY - This is the final note written by the support analyst in charge of this case. Sometimes, this can contain details about the case iteself, but oftentimes, it just contains an acknowledgement of whether the case was solved or not
</TAG INFO>
"""

key_facets = """Extract a short sentence succintly describing this case. Focus on extracting thesse aspects/facets about the case

<KEY FACETS>
- which module and feature of the product this support case was related to. Dont mention the high level module of Ross. Focus on one level below.
- what type of issue was this support case about - bugs, clarification, custom implementation requests, feature requests, etc
</KEY FACETS>

The examples provided for each facet are to just illustrate what they mean. They do not fully capture all the different values
that facet can have.

The sentence you extract will later be used to cluster support cases that belong to similar products/features and have similar
types of issues together. So, make sure that your sentence is information dense with respect to the given facets.
"""

prompt_template = """You are an expert analyst in charging of analysing and summarizing a support casese encountered by Aptean's Ross ERP Product.
You are given a set of highly unstructured notes related to this support case. Communication about the case from Aptean's customer
and internal notes/communication on resolving the issue by Aptean's internal team.

Your task is to extract summarize this information succintly in accordance to the themes/facets provided to you. Your summarization
of this case will be later used to figure out which cateogry of support cases to put this case into. So, keep your summary really
information dense.

<TASK DESCRIPTION>
{key_facets}
Don't include any customer specific information. Only focus on the given facets
</TASK DESCRIPTION>

<CASE INFORMATION SCHEMA DESCRIPTION>
{tag_info}
</CASE INFORMATION SCHEMA DESCRIPTION>

Here is the actual case information
<CASE INFORMATION>
{case_info}
</CASE INFORMATION>


Only respond with the summary. Dont add prefix or suffix text to explain
what you are responding with.
"""



def get_case_summary(case_info_dirpath, row):
    # case_number = row['Case Number']
    # case_assets_dirpath = case_info_dirpath.joinpath(f"{case_number}")
    # case_assets_dirpath.mkdir()

    # get case info
    case_info = get_case_info(row)
    # case_info_filepath = case_assets_dirpath.joinpath(case_info_filename)
    # with open(case_info_filepath, 'w') as f:
    #     f.write(case_info)


    # get case summary
    prompt = prompt_template.format(key_facets=key_facets, tag_info=tag_info, case_info=case_info)
    model_summary_response = make_openrouter_request(prompt)
    # model_summary_filepath = case_assets_dirpath.joinpath(model_summary_filename)
    # with open(model_summary_filepath, 'w') as f:
    #     f.write(model_summary_response)

    return model_summary_response

vo = voyageai.Client()
def embed_case_summary(case_summary_path):
    with open(case_summary_path) as f:
        case_summary = f.read()
    # print(case_summary)
    # print("-" * 25)
    # print("\n\n")
    embed_result = vo.embed(
        case_summary,
        model="voyage-3.5",
        input_type="document"
    )
    embedding = embed_result.embeddings[0]
    return embedding

genai_client = genai.Client()
def gemini_embed_case_summary(case_summary_path):
    with open(case_summary_path) as f:
        case_summary = f.read()

    embedding_response = genai_client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=case_summary,
        config=types.EmbedContentConfig(task_type="CLUSTERING")
    )

    embedding = embedding_response.embeddings[0].values

    return embedding

cohere_client = cohere.ClientV2("mtmAG2Bk2OU60NxWS1Xx2xiR8tiT3mvkqZsSALiD")
def cohere_embed_case_summary(case_summary_path):
    with open(case_summary_path) as f:
        case_summary = f.read()

    print(case_summary)
    print("-" * 25)
    print("\n\n")

    embedding_response = cohere_client.embed(
        # model="embed-v4.0",
        model="embed-english-v3.0",
        input_type="search_document",
        texts=[case_summary],
        embedding_types=["float"],
    )

    embedding = np.array(embedding_response.embeddings.float[0])

    return embedding


# sample_path_1 = "/Users/suryakrishnan/Documents/GitHub/aptean/excel_experiment/case_info/3432670/model_summary.md"
# sample_path_2 = "/Users/suryakrishnan/Documents/GitHub/aptean/excel_experiment/case_info/3994581/model_summary.md"
# # sample_path_2 = "/Users/suryakrishnan/Documents/GitHub/aptean/excel_experiment/case_info/2686339/model_summary.md"

# sample_embedding_1 = cohere_embed_case_summary(sample_path_1)
# sample_embedding_2 = cohere_embed_case_summary(sample_path_2)

# sample_embedding_1 = np.expand_dims(sample_embedding_1, 0)
# sample_embedding_2 = np.expand_dims(sample_embedding_2, 0)
# print(sample_embedding_1.shape, sample_embedding_2.shape)


# similarity_scores = sklearn.metrics.pairwise.cosine_similarity(sample_embedding_1, sample_embedding_2, dense_output=True)
# print(similarity_scores.shape)
# print(similarity_scores)
# exit()


# filtered_df = df.iloc[:5000]

# # get all case summary

# case_summary_list = []
# partial_get_case_summary = partial(get_case_summary, case_info_dirpath)
# with ThreadPoolExecutor(max_workers=42) as executor:
#     # Start the load operations and mark each future with its URL
#     case_summary_futures = [executor.submit(partial_get_case_summary, row) for _, row in filtered_df.iterrows()]
#     for future in tqdm(as_completed(case_summary_futures), total=len(case_summary_futures)):
#         case_summary = future.result()
#         case_summary_list.append(case_summary)

# print(len(case_summary_list))

# with open(case_summary_list_filepath, 'w') as f:
#     json.dump(case_summary_list, f, indent=4)

# case_number_list = [row['Case Number'] for _, row in filtered_df.iterrows()]
# with open(case_number_filepath, 'w') as f:
#     json.dump(case_number_list, f, indent=4)



# # embed all case summaries
# from sentence_transformers import SentenceTransformer

# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# case_summary_embeddings = []
# for case_summary in tqdm(case_summary_list):
#     case_summary_embedding = embedder.encode([case_summary])[0]
#     case_summary_embeddings.append(case_summary_embedding)

# case_summary_embeddings = np.array(case_summary_embeddings)

# np.save(summary_embedding_filepath, case_summary_embeddings)



# # test to read some similarities
# for embedding in case_summary_embeddings[0:]:
#     embedding = np.expand_dims(embedding, 0)
#     similarity_scores = sklearn.metrics.pairwise.cosine_similarity(embedding, case_summary_embeddings, dense_output=True)

#     sorted_idx_list = np.argsort(similarity_scores)[:, ::-1]
#     sorted_scores = np.take_along_axis(similarity_scores, sorted_idx_list, 1)
#     print(sorted_scores)

#     top_sorted_idx_list = sorted_idx_list[0, :10]

#     for idx in top_sorted_idx_list:
#         print('similar', case_number_list[idx], similarity_scores[0, idx])
#         case_summary = case_summary_list[idx]
#         print(case_summary)
#         print("\n\n")

#     break



# # load embeddings
# case_summary_embeddings = np.load(summary_embedding_filepath)
# with open(case_summary_list_filepath) as f:
#     case_summary_list = json.load(f)

# print(len(case_summary_embeddings))
# print(len(case_summary_list))


# # Perform clustering
# clusterer = sklearn.cluster.KMeans(n_clusters=30)
# # clusterer = sklearn.cluster.HDBSCAN(min_cluster_size=50)
# start = time.time()
# cluster_predictions = clusterer.fit_predict(case_summary_embeddings)
# end = time.time()
# print(f"time spend on clustering", end - start)
# print(cluster_predictions.shape)





# Extract cluster topics

# cluster_topic_analysis_prompt_template = """You are a support analyst working at aptean. you are given a subset of related support case
# summaries that have a similar pattern and your task is to come up with the topic for this cluster/pattern of cases. Provide your detailed
# analysis on what a descriptive topic for this cluster should be


# <DATA SCHEMA EXPLANATION>
# - a line of `-` characters act as the separator between cases part of this cluster
# - The cases usually talk about bugs or new feature requests. go more specific in your analysis than simply saying bugs/feature requests in several modules in Ross ERP product
# - List the common pattern in the bug they are reporting or the feature they are requesting specifically.
# </DATA SCHEMA EXPLANATION>


# <INSTRUCTIONS>
# - Try to be as specific as possible.
# - It only represents a subset of Ross ERP support cases. So, avoid saying something generic like Ross ERP support cases or things like features, bugs across various modules in the Ross ERP product.
# - Focus on what module of the Ross product the support cases fall under and what type/pattern of issues are represented in the support cases.
# - Your topic sentence should represent the broad functional area (e.g., Access, Reporting, Performance, etc), Specific module, workflow, or component and the concrete failure pattern
# </INSTRUCTIONS>

# <CASES INFO>
# {cases_info}
# </CASES INFO>
# """

# cluster_topic_prompt_template = """Given an analysis for a cluster topic name. Come up with a short, descriptive name to
# represent the cluster. Make it as specific and descriptive as posibble

# - Respond with just the cluster topic. 
# - Do not include any analysis, explanation or filler words

# <CLUSTER TOPIC ANALYSIS>
# {cluster_topic_analysis}
# </CLUSTER TOPIC ANALYSIS>
# """


# filtered_df['cluster'] = cluster_predictions
# filtered_df_group_by = filtered_df.groupby('cluster')

# cluster_id2topic_analysis = {}
# for cluster_id, cluster_df in filtered_df_group_by:
#     summary_string = ""
#     separator = "-" * 25
#     print(len(cluster_df))
#     row_count = 0
#     for row_idx, row in cluster_df.iterrows():
#         case_summary = case_summary_list[row_idx]
#         summary_string += f"{case_summary}\n{separator}\n\n"

#         if row_count >= 20:
#           break

#         row_count += 1
#     # print(len(summary_string))

#     cluster_topic_analysis_prompt = cluster_topic_analysis_prompt_template.format(cases_info=summary_string)
#     cluster_topic_analysis = make_openrouter_request(cluster_topic_analysis_prompt)
#     cluster_id2topic_analysis[cluster_id] = cluster_topic_analysis

#     # print(cluster_topic_analysis)
#     # print("-" * 25)
#     # print("\n\n")

# cluster_id2topic = {}
# for cluster_id, cluster_topic_analysis in cluster_id2topic_analysis.items():
#   cluster_topic_prompt = cluster_topic_prompt_template.format(cluster_topic_analysis=cluster_topic_analysis)
#   cluster_topic = make_openrouter_request(cluster_topic_prompt)
#   cluster_id2topic[cluster_id] = cluster_topic

#   print(cluster_topic)
#   print("-" * 25)
#   print("\n\n")


# cluster_topic_str = ""
# for topic in cluster_id2topic.values():
#     separator = "-" * 25
#     cluster_topic_str += f"{topic}\n{separator}\n\n"

# with open(cluster_topics_filepath, 'w') as f:
#     f.write(cluster_topic_str)



# temp_data = {
#     "cluster": cluster_predictions.tolist(),
#     "cluster_id2topic": cluster_id2topic
# }
# with open('temp.json', 'w') as f:
#     json.dump(temp_data, f, indent=4)


# with open('temp.json') as f:
#     temp_data = json.load(f)
# cluster = temp_data["cluster"]
# cluster_id2topic = temp_data["cluster_id2topic"]
# filtered_df["cluster"] = cluster

# with open(case_summary_list_filepath) as f:
#     case_summary_list = json.load(f)
# case_summary_list = np.array(case_summary_list)


# example_cases_per_topic = 10
# cluster_id2info = {}
# for cluster_id, cluster_topic in cluster_id2topic.items():
#     cluster_df = filtered_df.loc[filtered_df['cluster'] == int(cluster_id)]

#     # get a few exmple summary
#     sampled_cluster_df = cluster_df.sample(example_cases_per_topic)
#     sample_index_list = sampled_cluster_df.index.tolist()

#     sample_case_summary_list = case_summary_list[sample_index_list]
    

#     separator_line = "-" * 25
#     case_divider = f"\n{separator_line}\n\n"

#     sample_case_str = case_divider.join((sample_case_summary_list))
#     cluster_info = f"## Cluster Topic\n\n{cluster_topic}\n\n## Example Case Summaries in this Cluster\n\n{sample_case_str}\n"

#     cluster_id2info[cluster_id] = cluster_info


# overall_str = ""
# for cluster_info in cluster_id2info.values():
#     overall_str += f"{cluster_info}\n\n\n\n"
# with open('cluster_info.md', 'w') as f:
#     f.write(overall_str)
    

# merge topics if they can be merged
    # give a bunch of topics and example cases - go over one by one and see if any of them can be
    # merged?
        # merge
        # use the second one and only go forward and see if any can be merge 


# merge_cluster_prompt_template = """You are an expert analysing two clusters/sets of support cases and are tasked with deciding
# whether these two clusters/sets should be merged into the same cluster/set or not.

# The criteria you should use to deciding on whether the two clusters of support case should be merged into a single set of 
# support case is whether both sets of support cases involve the 

#  - same subset of features/modules 
#  - same pattern of issues in their features/modules.

# If the pattern of cases in both the clusters are the same with respect to these aspects, conclude that they should be merged into
# one, if not, say they should not be merged into one

# <CLUSTER NO 1 INFORMATION>
# {cluster_1_info}
# </CLUSTER NO 1 INFORMATION>

# <CLUSTER NO 2 INFORMATION>
# {cluster_2_info}
# </CLUSTER NO 2 INFORMATION>

# Only repond with the word yes/no on whether these two clusters need to be merged.

# """

# class MergeClusterAnalysis(BaseModel):
#     """Analysis and conclusion on whether the two given clusters should be merged into one according
#     to the given criteria"""

#     reasoning: str = Field(..., description="Reasoning behind your decision on whether the two given clusters should be merged into one or not")
#     should_be_merged: bool = Field(..., description="Should the given two clusters be merged into one or not")

# adaptor = PydanticAdaptorOpenRouter()

# # iterate through find yes, merge build one
# cluster_info_items = list(cluster_id2info.items())
# merged_cluster_id_list = []
# while len(cluster_info_items) > 0:
#     print(len(cluster_info_items))
#     query_cluster_id, query_cluster_info = cluster_info_items[0]
#     curr_merge_cluster_id_list = [query_cluster_id]
#     if len(cluster_info_items) > 1:
#         for idx in tqdm(range(1, len(cluster_info_items)), total=len(cluster_info_items)-1):
#             curr_cluster_id, curr_cluster_info = cluster_info_items[idx]
#             merge_cluster_prompt = merge_cluster_prompt_template.format(cluster_1_info=query_cluster_info, cluster_2_info=curr_cluster_info)

#             content = [{"type": "text", "text": merge_cluster_prompt}]
#             curr_message = {
#                 "role": "user",
#                 "content": content
#             }
#             message_history = [curr_message]
#             merge_cluster_analysis = adaptor.chat.completions.create(
#                 pydantic_model=MergeClusterAnalysis,
#                 num_retries=5,
#                 model="openai/gpt-4o-mini",
#                 messages=message_history,
#                 max_tokens=4096,
#                 stream=False
#             )

#             if merge_cluster_analysis.should_be_merged:
#                 curr_merge_cluster_id_list.append(curr_cluster_id)
    
#     # get all the cluster ids and add them
#     cluster_info_items = [(cluster_id, cluster_info) for cluster_id, cluster_info in cluster_info_items if cluster_id not in curr_merge_cluster_id_list]
#     merged_cluster_id_list.append(curr_merge_cluster_id_list)


# print(merged_cluster_id_list)    


# sum = 0
# for item in merged_cluster_id_list:
#     sum += len(item)
# print(sum)

# filtered_df["merged_cluster"] = None
# for curr_cluster_group in merged_cluster_id_list:
#     curr_cluster_group = [str(cluster_id) for cluster_id in curr_cluster_group]
#     selected_rows = filtered_df["cluster"].isin(curr_cluster_group)
#     filtered_df.loc[selected_rows]["merged_cluster"] = curr_cluster_group[0]

# print(filtered_df["merged_cluster"].value_counts())

