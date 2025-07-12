from typing import List, Literal
import json
import time
from functools import partial

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



openai_client = OpenAI()
openai_adaptor = PydanticAdaptorOpenRouter(openai_client=openai_client)



openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    # api_key=os.getenv('OPENROUTER_API_KEY')
)
openrouter_adaptor = PydanticAdaptorOpenRouter(openai_client=openrouter_client)
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
- what was the actual solution that lead to the resolution of the case. Make it succint and thorough. Provide enough details about the resolution
so a new support agent can know exactly how to solve if a similar issue comes up in the future.
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
    # # ROSS
    # path = "/workspace/aptean/ROSS_case_list.xlsx"

    # df = pd.read_excel(path)
    # filtered_df = df.iloc[:100]

    # case_summary_df = add_case_summary_df(filtered_df)

    # case_cluster_df = cluster_case_summaries(case_summary_df, n_clusters=3)

    # cluster_topics_df = extract_cluster_topics(case_cluster_df)

    # deduplicated_cluster_df = deduplicate_clusters(cluster_topics_df)

    support_case_clustering_prompt_template = """You are given a set of support case summaries for an Aptean ERP product. The support case
    summaries all succinctly talk about what the issue is. Your task is to thoroughly analyze all the support case summaries and group them into
    sets where each set of support case summaries can all be solved by the same solution. Then, you should write the solution that will solve all these
    support cases and prevent them in the future.

    <SOLUTION TYPE INFORMATION>
    Here are the common types of solutions to support cases
    
    - Feature Addition
    - Bug fix
    - Add Knowledgebase article
    - Automate an administrative task

    Multiple support cases should be grouped together if the same solution can solve all of them and prevent them from occurring in the future. Same
    solution is represented by the following ideas.

        - If the solution is a feature addition, then adding the same exact feature will solve the same set of support issues. These features
        should match the specificity discussed in the support cases
        -  If the solution is a bug fix, then fixing the same exact bug will solve the same set of support issues. The bug fixes should match the
        specificity discussed in the support cases
        - If the solution is a knowledgebase article, then adding a knowledgebase article with the same exact topic will solve the same set of 
        support issues. Think about what the scope of a self-sustaining digestible knowledgebase is and use that as a metric as well as try to match
        the specificity discussed in the support cases.
        - If the solution is automating an administrative task, then automating the exact administrative task will solve the same set of support issues.
        Think about what the scope of each automatable administrative task is and use that as a metric as well as try to match the specificity discussed
        in the support cases.

    Notes:
        - Administrative tasks are not administrative tasks inside the product, but rather refer to administraive tasks that have to be done by the Aptean support team
        to solve certain issues
    </SOLUTION TYPE INFORMATION>

    <TASK INFORMATION>
    Your task is to analyze all the summaries of support cases and group them into sets where each set of support case summaries can all be solved by the
    same solution and then write the solution that will solve all these support cases and prevent them in the future. Look at the above mentioned information
    for some detail on what are the different solution types and what a `same solution` means. Be extremely specific and thorough in your solution

    Instructions
    - Group sets of support cases together if and only if they will all be solved by the same solution.
    - Make sure to not ignore any support case summary. Every support case should be accounted for.
    - If a support case is unique and a solution only solves that support case. Create a group with just that support case in it and add the solution to solve it.
    - Every support case in the given <SUPPORT CASE SUMMARIES> should be put in a group.
    - Each support case in the given <SUPPORT CASE SUMMARIES> can only be put in a group once.
    - Make sure to keep the support case ID exactly the same.
    
    Notes:
        - If a functionality already exists in the product, but the user is not aware of it, then suggest adding a knowledgebase article to inform the user about it.

    For each group of support cases
        - write your reasoning behind grouping all the support cases in this group together according to the above given instructions
        - write your summary of the support cases represented by this group
        - list down the id values of all the support cases that are part of this group and will be solved by the same solution.
        - Describe the type of solution that will solve all the support cases in this grouping
        - Describe the actual solution that needs to be implemented to solve all the support cases in this group and prevent them in the future. Make sure that your solution type is extremely thorough, clear and complete.
            * If you are suggesting a feature addition, Describe the feature that needs to be added in detail.
            * If you are suggesting a bug fix, describe in detail what bug needs to be fixed and if its clear how, mention that as well.
            * If you are suggesting a knowledgebase article, describe in detail what the knowledgebase article needs to be about in detail. If possible, describe a suitable title and a description for the knowledgebase article.
            * If you are suggesting automating an administrative task, describe in detail what administrative task needs to be automated and if possible, describe how to efficiently automate it as well.
    </TASK INFORMATION>

    Here are all the support case summaries that you should group and solve
    <SUPPORT CASE SUMMARIES>
    {support_case_summary_list}
    </SUPPORT CASE SUMMARIES>
    """

    class SupportCaseGroupInfo(BaseModel):
        """This data model captures the information on each group of support cases"""
        reasoning: str = Field(..., description="What is the reasoning behind grouping all these support cases together according to the given instructions")
        summary: str = Field(..., description="combined summary of the support cases represented by this group. Leave out support case ids out of it. This summary will be used as the topic sentence to describe this cluster")
        support_case_id_list: List[str] = Field(..., description="List of support cases that are part of this group and will be all solved by the same solution")
        solution_type: Literal['feature_addition', 'bug_fix', 'knowledgebase_article', 'automate_administrative_task'] = Field(..., description="The type of solution that will solve all the support cases in this grouping")
        solution_description: str = Field(..., description="Describe the actual solution that needs to be implemented according to the solution type. Make sure that your solution type is extremely thorough and complete.")

    class SupportCaseClustering(BaseModel):
        """This data model captures the result of clustering all the different support cases into individual groups where all the cases in each group can be 
        solved by the same solution"""

        group_list: List[SupportCaseGroupInfo] = Field(..., description="list of groups of support cases that can all be solved by the same solution")


    support_case_group_clustering_prompt_template = """You are given information on a bunch of support case groups for an Aptean ERP product. Each support case group
    contains a summary succinctly describing the set of support cases it represents and describes a solution that will solve the set of support cases it represents and
    prevent them from occurring in the future. 

    Your task is to thoroughly analyze all the information for each support case group and cluster them into sets where the set of support case groups all represent the 
    same type of support cases and can all be solved by the same solution. Then, you should write the solution that will solve all the support cases represented by this
    set of support case groups and prevent them from occurring in the future.

    <SOLUTION TYPE INFORMATION>
    Here are the common types of solutions to support cases
    
    - Feature Addition
    - Bug fix
    - Add Knowledgebase article
    - Automate an administrative task

    Multiple support case groups should be grouped together if the same solution can solve all of them and prevent them from occurring in the future. Same
    solution is represented by the following ideas.

        - If the solution is a feature addition, then adding the same exact feature will solve the same set of support issues. These features
        should match the specificity discussed in the support cases
        -  If the solution is a bug fix, then fixing the same exact bug will solve the same set of support issues. The bug fixes should match the
        specificity discussed in the support cases
        - If the solution is a knowledgebase article, then adding a knowledgebase article with the same exact topic will solve the same set of 
        support issues. Think about what the scope of a self-sustaining digestible knowledgebase is and use that as a metric as well as try to match
        the specificity discussed in the support cases.
        - If the solution is automating an administrative task, then automating the exact administrative task will solve the same set of support issues.
        Think about what the scope of each automatable administrative task is and use that as a metric as well as try to match the specificity discussed
        in the support cases.
    </SOLUTION TYPE INFORMATION>

    <TASK INFORMATION>
    Your task is to analyze all the information on each support case group and group them into sets where each set of support case groups can all be solved by the
    same solution and then write the solution that will solve all these support cases and prevent them in the future. Look at the above mentioned information
    for some detail on what are the different solution types and what a `same solution` means and use the solution part of each individual support case group
    in the set as your inspiration. If they already solve the problem well enough, then use them as is. If they need to be modified to accommodate all the cases
    in the combined groups of support case groups, then do the necessary modification. Be extremely specific and thorough in your solution.

    Instructions
    - Group sets of support case groups together if and only if they will all be solved by the same solution.
    - Make sure to not ignore any support case group. Every support case group should be accounted for.
    - If a support case group is unique and a solution only solves that support case group. Create a group with just that support case group in it and add the solution for it.
    - Every support case group in the given <SUPPORT CASE GROUP INFO LIST> should be put in a group.
    - Each support case group in the given <SUPPORT CASE GROUP INFO LIST> can only be put in a group once.
    - Make sure to keep the support case group ID exactly the same.

    For each group of support case groups
        - write your reasoning behind grouping all the support case groups in this group together according to the above given instructions
        - write your summary of the support case groups represented by this meta group
        - list down the id values of all the support case groups that are part of this group and will be solved by the same solution.
        - Describe the type of solution that will solve all the support cases in this grouping
        - Describe the actual solution that needs to be implemented to solve all the support case groups in this group and prevent them in the future. Make sure that your solution type is extremely thorough, clear and complete.
            * If you are suggesting a feature addition, Describe the feature that needs to be added in detail.
            * If you are suggesting a bug fix, describe in detail what bug needs to be fixed and if its clear how, mention that as well.
            * If you are suggesting a knowledgebase article, describe in detail what the knowledgebase article needs to be about in detail. If possible, describe a suitable title and a description for the knowledgebase article.
            * If you are suggesting automating an administrative task, describe in detail what administrative task needs to be automated and if possible, describe how to efficiently automate it as well.
    </TASK INFORMATION>

    Here are all the support case group information that you should group and solve
    <SUPPORT CASE GROUP INFO LIST>
    {support_case_group_info_list}
    </SUPPORT CASE GROUP INFO LIST>
    """
    class SupportCaseGroupClustering(BaseModel):
        """This data model captures the result of clustering all the different support case groups into sets where all the support case groups in each meta group can be 
        solved by the same solution"""

        group_list: List[SupportCaseGroupInfo] = Field(..., description="list of groups of support case groups that can all be solved by the same solution")





    def gen_two_pass_structured_llm_response(prompt, pydantic_model_class, message_history=None):
        if message_history is None:
            message_history = []
        
        initial_user_message = {
            "role": "user",
            "content":  [{"type": "text", "text": prompt}]
        }
        message_history.append(initial_user_message)
        start = time.time()
        print('getting raw text')
        #TODO try and use a cheap/powerful reasoning model here. Try to get close to o1 level performance -> Deepseek reasoning model maybe ?
        raw_text_response = openai_client.chat.completions.create(
            # model="o1",
            # model="meta-llama/llama-4-maverick",
            model="gpt-4o-mini",
            # model="o3-mini",
            messages=message_history,
            # max_tokens=4096,
            stream=False
        )
        print('got raw text', time.time() - start)
        print('raw text usage', raw_text_response.usage)
        raw_text = raw_text_response.choices[0].message.content
        assistant_message = {
            "role": "assistant",
            "content": raw_text
        }

        format_prompt = f"""format the extracted grouping data in the given data model. Be extremely thorough and accurate.
        Here is the content to format
        """
        format_user_message = {
            "role": "user",
            "content": [{"type": "text", "text": format_prompt}]
        }
        message_history.append(assistant_message)
        message_history.append(format_user_message)
        start = time.time()
        print('formatting raw text')
        # TODO use a better model here as well. Find a cheaper/powerful alternative. Try to get close to o1 level performance
        pydantic_data_model = openai_adaptor.chat.completions.create(
            pydantic_model=pydantic_model_class,
            num_retries=1,
            # model="gpt-4o",
            # model="meta-llama/llama-4-maverick",
            model="gpt-4o-mini",
            # model="o1",
            messages=message_history,
            # max_tokens=4096,s
            stream=False
        )
        print('formatted raw text', time.time() - start)
                
        return (pydantic_data_model, message_history)

    
    def get_validated_structured_response(prompt, pydantic_model_class, model_validator, num_retries=0):
        pydantic_data_model, message_history = gen_two_pass_structured_llm_response(prompt, pydantic_model_class)
        is_valid, error_str = model_validator(pydantic_data_model)
        retry_count = 0
        while not is_valid and retry_count < num_retries:
            retry_count += 1
            print(f"retrying - {retry_count}")
            print("This is the reported error", error_str)

            error_user_message = {
                "role": "user",
                "content": [{"type": "text", "text": f"Fix these errors:\n\n{error_str}"}]
            }
            message_history.append(error_user_message)
    
            pydantic_data_model, message_history = gen_two_pass_structured_llm_response(prompt, pydantic_model_class, message_history)
            is_valid, error_str = model_validator(pydantic_data_model)
        

        #TODO BUG we should only be returining the data model as output if it satisifies all validation logic. 
        # UNCOMMENTING for now as we dont have any error handling or null checks later.
        if (is_valid):
            return pydantic_data_model
        else:
            return None
        
        # HIGHLY NOT RECOMMEND -> 
        # return pydantic_data_model


    def get_initial_support_case_groups(case_id_list, case_summary_list, chunk_size=100):
        case_divider = f"\n{'-' * 25}\n\n"
        total_case_count = len(case_summary_list)
        batch_case_clustering_list = []
        for i in range(0, total_case_count, chunk_size):
            print("batch start idx", i)
            batch_start_idx = i
            batch_end_idx = i + chunk_size

            batch_case_summary_list = case_summary_list[batch_start_idx: batch_end_idx]
            batch_case_info_list_str = ""
            for batch_item_idx, case_summary in enumerate(batch_case_summary_list):
                global_case_idx = batch_start_idx + batch_item_idx
                case_id = case_id_list[global_case_idx]

                case_info_str = f"**Case ID:**\n\n{case_id}\n\n**Case Summary:**\n\n{case_summary}\n\n{case_divider}"
                batch_case_info_list_str += case_info_str
            
            batch_case_clustering_prompt = support_case_clustering_prompt_template.format(support_case_summary_list=batch_case_info_list_str)
                        
            batch_case_id_list = case_id_list[batch_start_idx: batch_end_idx]
            partial_case_clustering_validator = partial(validate_case_clustering, case_id_list=batch_case_id_list)
            batch_case_clustering = get_validated_structured_response(batch_case_clustering_prompt, SupportCaseClustering, partial_case_clustering_validator, num_retries=10)
            
            if not batch_case_clustering:
                raise ValueError(f'could not cluster the batch. Batch start idx - {batch_start_idx} and end idx - {batch_end_idx}')

            batch_case_clustering_list.append(batch_case_clustering)

        group_info_dict = get_group_info_dict(batch_case_clustering_list)
        
        return group_info_dict

    
    def validate_case_clustering(case_clustering, case_id_list):
        case_id2group_idx_list = {}
        extra_case_id_list = []
        for group_idx, group_info in enumerate(case_clustering.group_list):
            for case_id in group_info.support_case_id_list:
                if case_id in case_id_list:
                    if case_id in case_id2group_idx_list:
                        case_id2group_idx_list[case_id].append(group_idx)
                    else:
                        case_id2group_idx_list[case_id] = [group_idx]
                else:
                    extra_case_id_list.append(case_id)

        multiple_group_case_id_list = []
        for case_id, group_idx_list in case_id2group_idx_list.items():
            if len(group_idx_list) > 1:
                multiple_group_case_id_list.append(case_id)

        missing_case_id_list = []
        for case_id in case_id_list:
            if case_id not in case_id2group_idx_list:
                missing_case_id_list.append(case_id)
        

        error_str = ""
        if len(extra_case_id_list) > 0:
            extra_case_error_str = f"These case ids do not exist:\n\n{extra_case_id_list}\n\n"
            error_str += extra_case_error_str
        if len(multiple_group_case_id_list) > 0:
            multiple_group_case_error_str = f"These cases belong to multiple groups. Each case should belong to only one group:\n\n{multiple_group_case_id_list}\n\n"
            error_str += multiple_group_case_error_str
        if len(missing_case_id_list) > 0:
            missing_case_id_str = f"These case ids have not been placed into a group at all:\n\n{missing_case_id_list}\n\n"
            error_str += missing_case_id_str

        if error_str:
            error_str = f"Look back the initial instructions and fix all these errors\n\n{error_str}"
            return False, error_str
        
        return True, None


    def get_group_info_dict(batch_case_clustering_list):
        group_info_dict = {}
        group_idx = 0
        for batch_case_clustering in batch_case_clustering_list:
            for support_case_group in batch_case_clustering.group_list:
                group_info_dict[group_idx] = support_case_group.model_dump()
                group_idx += 1

        return group_info_dict
    
    def get_case_id2group_info(group_info_dict):
        case_id2group_info = {}
        for group_idx, group_info in group_info_dict.items():
            for case_id in group_info["support_case_id_list"]:
                case_id2group_info[case_id] = group_info
        
        return case_id2group_info
    

    
    def add_case_cluster_info(df):
        cluster_start = time.time()
        
        df['case_cluster_summary'] = None
        df['case_cluster_reasoning'] = None
        df['case_cluster_solution_type'] = None
        df['case_cluster_solution_description'] = None

        # only ptocess big enough clusters
        if len(df) >= 10:  
            # get intial support case summary
            case_summary_list = df['case_summary'].tolist()
            case_number_list = [str(num) for num in df['Case Number'].tolist()]
            group_info_dict = get_initial_support_case_groups(case_number_list, case_summary_list, chunk_size=len(case_number_list))
            case_id2group_info = get_case_id2group_info(group_info_dict)

            print(f"num groups - {len(group_info_dict)}")

            for row_idx, row in df.iterrows():
                case_number = str(row["Case Number"])
                group_info = case_id2group_info[case_number]

                case_cluster_summary = group_info['summary']
                case_cluster_reasoning = group_info['reasoning']
                case_cluster_solution_type = group_info['solution_type']
                case_cluster_solution_description = group_info['solution_description']

                df.loc[row_idx, 'case_cluster_summary'] = case_cluster_summary
                df.loc[row_idx, 'case_cluster_reasoning'] = case_cluster_reasoning
                df.loc[row_idx, 'case_cluster_solution_type'] = case_cluster_solution_type
                df.loc[row_idx, 'case_cluster_solution_description'] = case_cluster_solution_description
            
        print(f"time taken for cluster {cluster_group_col_list} - {time.time() - cluster_start}")

        return df






    case_summary_df_path = "/workspace/aptean/scratchpad_data/case_summary.xlsx"
    case_summary_df = pd.read_excel(case_summary_df_path)

    #TODO BUG REMOVE THIS WHEN YOU EVENTUALLY FIND A MODEL THAT YOU ARE HAPPY WITH IN TERMS OF COST/PERFORMANCE
    case_summary_df = case_summary_df.iloc[:100]


    overall_start = time.time()
    groupby_df_list = []
    with ThreadPoolExecutor(max_workers=42) as executor:
        groupby_df_cluster_futures2col_list = {}
        for cluster_group_col_list, groupby_df in case_summary_df.groupby(['deduplicated_module_name', 'category_name']):            
            groupby_df_cluster_futures2col_list[executor.submit(add_case_cluster_info, groupby_df)] = cluster_group_col_list
        
        for future in tqdm(as_completed(groupby_df_cluster_futures2col_list), total=len(groupby_df_cluster_futures2col_list)):
            groupby_col_list = groupby_df_cluster_futures2col_list[future]
            groupby_case_cluster_df = future.result()
            
            groupby_df_list.append(groupby_case_cluster_df)
    print(f"time taken for all clusters - {time.time() - overall_start}")

    case_clustered_df = pd.concat(groupby_df_list)
    case_clustered_df.to_excel("/workspace/aptean/scratchpad_data/case_clustered.xlsx", index=False)
        
