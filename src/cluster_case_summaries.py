import pandas as pd
import numpy as np

import sklearn.cluster
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from openai import OpenAI
import voyageai
import voyageai.client
from google import genai
from google.genai import types
import cohere

import sklearn

from pydantic import BaseModel, Field
from PydanticAdaptorAnthropic import PydanticAdaptorAnthropic
from PydanticAdaptorOpenRouter import PydanticAdaptorOpenRouter

from sentence_transformers import SentenceTransformer

import os
from dotenv import load_dotenv

load_dotenv()

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

openrouter_client = OpenAI(
    # base_url="https://openrouter.ai/api/v1",
    # api_key=os.getenv('OPENROUTER_API_KEY')
)
def make_openrouter_request(prompt):    
    resp = openrouter_client.chat.completions.create(
        # model="meta-llama/llama-4-maverick",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        # temperature=0.3,
        # max_tokens=4000,
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


# Extract cluster topics
cluster_topic_analysis_prompt_template = """You are a support analyst working at aptean. you are given a subset of related support case
summaries that have a similar pattern and your task is to come up with the topic for this cluster/pattern of cases. Provide your detailed
analysis on what a descriptive topic for this cluster should be


<DATA SCHEMA EXPLANATION>
- a line of `-` characters act as the separator between cases part of this cluster
- The cases usually talk about bugs or new feature requests. go more specific in your analysis than simply saying bugs/feature requests in several modules in Ross ERP product
- List the common pattern in the bug they are reporting or the feature they are requesting specifically.
</DATA SCHEMA EXPLANATION>


<INSTRUCTIONS>
- Try to be as specific as possible.
- It only represents a subset of Ross ERP support cases. So, avoid saying something generic like Ross ERP support cases or things like features, bugs across various modules in the Ross ERP product.
- Focus on what module of the Ross product the support cases fall under and what type/pattern of issues are represented in the support cases.
- Your topic sentence should represent the broad functional area (e.g., Access, Reporting, Performance, etc), Specific module, workflow, or component and the concrete failure pattern
</INSTRUCTIONS>

<CASES INFO>
{cases_info}
</CASES INFO>
"""

cluster_topic_prompt_template = """Given an analysis for a cluster topic name. Come up with a short, descriptive name to
represent the cluster. Make it as specific and descriptive as posibble

- Respond with just the cluster topic. 
- Do not include any analysis, explanation or filler words

<CLUSTER TOPIC ANALYSIS>
{cluster_topic_analysis}
</CLUSTER TOPIC ANALYSIS>
"""


# Merge Clusters
merge_cluster_prompt_template = """You are an expert analysing two clusters/sets of support cases and are tasked with deciding
whether these two clusters/sets should be merged into the same cluster/set or not.

The criteria you should use to deciding on whether the two clusters of support case should be merged into a single set of 
support case is whether both sets of support cases involve the 

    - same subset of features/modules 
    - same pattern of issues in their features/modules.

If the pattern of cases in both the clusters are the same with respect to these aspects, conclude that they should be merged into
one, if not, say they should not be merged into one

<CLUSTER NO 1 INFORMATION>
{cluster_1_info}
</CLUSTER NO 1 INFORMATION>

<CLUSTER NO 2 INFORMATION>
{cluster_2_info}
</CLUSTER NO 2 INFORMATION>

Only repond with the word yes/no on whether these two clusters need to be merged.

"""

class MergeClusterAnalysis(BaseModel):
    """Analysis and conclusion on whether the two given clusters should be merged into one according
    to the given criteria"""

    reasoning: str = Field(..., description="Reasoning behind your decision on whether the two given clusters should be merged into one or not")
    should_be_merged: bool = Field(..., description="Should the given two clusters be merged into one or not")







def get_case_summary(row):
    # get case info
    case_info = get_case_info(row)

    # get case summary
    prompt = prompt_template.format(key_facets=key_facets, tag_info=tag_info, case_info=case_info)
    model_summary_response = make_openrouter_request(prompt)

    return model_summary_response

vo = voyageai.Client()
def embed_case_summary(case_summary_path):
    with open(case_summary_path) as f:
        case_summary = f.read()

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

cohere_client = cohere.ClientV2()
def cohere_embed_case_summary(case_summary_path):
    with open(case_summary_path) as f:
        case_summary = f.read()

    embedding_response = cohere_client.embed(
        # model="embed-v4.0",
        model="embed-english-v3.0",
        input_type="search_document",
        texts=[case_summary],
        embedding_types=["float"],
    )

    embedding = np.array(embedding_response.embeddings.float[0])

    return embedding


def add_case_summary_df(df):
    # get all case summary
    case_summary_list = []
    row_idx_list = []
    print('adding all case summary')
    with ThreadPoolExecutor(max_workers=42) as executor:
        case_summary_futures2idx = {executor.submit(get_case_summary, row): row_idx for row_idx, row in df.iterrows()}
        
        for future in tqdm(as_completed(case_summary_futures2idx), total=len(case_summary_futures2idx)):
            row_idx = case_summary_futures2idx[future]
            case_summary = future.result()
            
            row_idx_list.append(row_idx)
            case_summary_list.append(case_summary)

    df.loc[row_idx_list, 'case_summary'] = case_summary_list
    return df


def embed_case_summaries(case_summary_df):
    case_summary_list = case_summary_df['case_summary'].tolist()

    # embed all case summaries
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    case_summary_embeddings = []
    print("embedding all case summary")
    for case_summary in tqdm(case_summary_list):
        case_summary_embedding = embedder.encode([case_summary])[0]
        case_summary_embeddings.append(case_summary_embedding)

    case_summary_embeddings = np.array(case_summary_embeddings)

    return case_summary_embeddings


def cluster_case_summaries(case_summary_df, n_clusters):

    case_summary_embeddings = embed_case_summaries(case_summary_df)

    # Perform clustering
    clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters)
    # clusterer = sklearn.cluster.HDBSCAN(min_cluster_size=50)
    cluster_predictions = clusterer.fit_predict(case_summary_embeddings)

    case_summary_df['cluster'] = cluster_predictions

    return case_summary_df


def extract_cluster_topics(case_cluster_df):
    case_cluster_df_groupby = case_cluster_df.groupby('cluster')

    cluster_id2topic_analysis = {}
    separator_line = "-" * 25
    case_divider = f"\n{separator_line}\n\n"
    num_example_cases_per_cluster = 25
    for cluster_id, cluster_df in case_cluster_df_groupby:
        # get some example case summaries belonging to this cluster
        sampled_cluster_df = cluster_df.sample(min(num_example_cases_per_cluster, len(cluster_df)))
        example_case_summary_list = sampled_cluster_df['case_summary'].tolist()
        example_cluster_cases_str = case_divider.join(example_case_summary_list)

        cluster_topic_analysis_prompt = cluster_topic_analysis_prompt_template.format(cases_info=example_cluster_cases_str)
        cluster_topic_analysis = make_openrouter_request(cluster_topic_analysis_prompt)
        cluster_id2topic_analysis[cluster_id] = cluster_topic_analysis


    cluster_id2topic = {}
    for cluster_id, cluster_topic_analysis in cluster_id2topic_analysis.items():
        cluster_topic_prompt = cluster_topic_prompt_template.format(cluster_topic_analysis=cluster_topic_analysis)
        cluster_topic = make_openrouter_request(cluster_topic_prompt)
        cluster_id2topic[cluster_id] = cluster_topic
    

    cluster_topic_list = []
    cluster_topic_analysis_list = []
    for _, row in case_cluster_df.iterrows():
        cluster_id = row["cluster"]

        cluster_topic_analysis = cluster_id2topic_analysis[cluster_id]
        cluster_topic = cluster_id2topic[cluster_id]
        
        cluster_topic_list.append(cluster_topic)
        cluster_topic_analysis_list.append(cluster_topic_analysis)

    case_cluster_df["cluster_topic"] = cluster_topic_list
    case_cluster_df["cluster_topic_analysis"] = cluster_topic_analysis_list

    return case_cluster_df


def deduplicate_clusters(cluster_topic_df):
    cluster_topic_df_groupby = cluster_topic_df.groupby('cluster')
    
    # Get all cluster info
    cluster_info_list = []
    num_example_cases_per_cluster = 10
    separator_line = "-" * 25
    case_divider = f"\n{separator_line}\n\n"
    for cluster_id, cluster_df in cluster_topic_df_groupby:

        cluster_topic = cluster_df['cluster_topic'].iloc[0]
        
        # get a few exmple summary
        sampled_cluster_df = cluster_df.sample(min(num_example_cases_per_cluster, len(cluster_df)))
        example_case_summary_list = sampled_cluster_df['case_summary'].tolist()
        example_cluster_cases_str = case_divider.join(example_case_summary_list)

        cluster_info = f"## Cluster Topic\n\n{cluster_topic}\n\n## Example Case Summaries in this Cluster\n\n{example_cluster_cases_str}\n"

        cluster_info_list.append((cluster_id, cluster_info))
    

    # dedpulicate similar clusters
    openai_client = OpenAI()
    adaptor = PydanticAdaptorOpenRouter(openai_client=openai_client)
    cluster_id_groupings_list = []
    print('starting deduplication')
    print('initial custer number', len(cluster_info_list))
    while len(cluster_info_list) > 0:
        query_cluster_id, query_cluster_info = cluster_info_list[0]
        curr_cluster_id_grouping = [query_cluster_id]
        
        if len(cluster_info_list) > 1:
            for idx in tqdm(range(1, len(cluster_info_list)), total=len(cluster_info_list)-1):
                curr_cluster_id, curr_cluster_info = cluster_info_list[idx]
                merge_cluster_prompt = merge_cluster_prompt_template.format(cluster_1_info=query_cluster_info, cluster_2_info=curr_cluster_info)

                content = [{"type": "text", "text": merge_cluster_prompt}]
                curr_message = {
                    "role": "user",
                    "content": content
                }
                message_history = [curr_message]
                merge_cluster_analysis = adaptor.chat.completions.create(
                    pydantic_model=MergeClusterAnalysis,
                    num_retries=5,
                    model="gpt-4o-mini",
                    messages=message_history,
                    max_tokens=4096,
                    stream=False
                )

                if merge_cluster_analysis.should_be_merged:
                    curr_cluster_id_grouping.append(curr_cluster_id)
        
        # get all the cluster ids and add them
        cluster_info_list = [(cluster_id, cluster_info) for cluster_id, cluster_info in cluster_info_list if cluster_id not in curr_cluster_id_grouping]
        cluster_id_groupings_list.append(curr_cluster_id_grouping)

        print('cluster number', len(cluster_info_list))

    
    cluster_topic_df["merged_cluster"] = None
    for curr_cluster_id_grouping in cluster_id_groupings_list:
        curr_cluster_id_grouping = [str(cluster_id) for cluster_id in curr_cluster_id_grouping]
        selected_rows = cluster_topic_df["cluster"].isin(curr_cluster_id_grouping)
        cluster_topic_df.loc[selected_rows]["merged_cluster"] = curr_cluster_id_grouping[0]

    return cluster_topic_df



if __name__ == "__main__":
    # ROSS
    path = "/workspace/aptean/ROSS_case_list.xlsx"

    df = pd.read_excel(path)
    filtered_df = df.iloc[:100]

    case_summary_df = add_case_summary_df(filtered_df)

    case_cluster_df = cluster_case_summaries(case_summary_df, n_clusters=3)

    cluster_topics_df = extract_cluster_topics(case_cluster_df)

    deduplicated_cluster_df = deduplicate_clusters(cluster_topics_df)