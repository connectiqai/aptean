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

from typing import Literal, Optional
from scratchpad import get_case_info


categories_info = """
    Here are the error categories: 

    - Integration: If an issue involves either an integration feature - whether its a bug, a general issue,
    feature or any other type of issue, classify it as an integration

    - Technical Error: If an issue directly involves a technical bug and is not related to integrations, then 
    classify it as a technical error. Some examples of technical bugs are things like things freezing, things being too slow, 
    features not working as intended, etc.

    - Feature Enhancement: If an issue is directly related to a module or a set of modules and involves the addition of a
    new feature to the product, then classify it as a feature enhancement.

    - Module Issue: If an issue is directly related to a module or a set of modules and is not related to integrations, 
    a technical bug, or a feature enhancement, then classify it as a module issue. 

    - Admin Issue: If an issue is involves admin related stuff such as adding users, removing users, ip whitelisting
    or other admin related tasks, then classify them as an admin issue.

    - Other: If an issue does not follow in any of these categories. Classify it as other. When you classify an issue
    as other. Provide a name for a category that should be created that will represent the class of issues represented
    by this current issue. Only do this if the issue cannot be clearly placed in any of the above mentioned categories.
    Also, provide reasoning behinf why it could not be placed in any of the above mentioned categories and why the suggested
    name is a good category name for the class of issues represented by the current issue
"""


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

prompt_template = """You are an expert analyst in charging of analysing and summarizing a support cases encountered by Aptean's customer support team.
You are given a set of highly unstructured notes related to this support case and communication about the case from Aptean's customer
and internal notes/communication on resolving the issue by Aptean's internal team.

Your task is to categorize this issue accordding to the categories provided to you. Only select one category per case and be 
extremely thoughtful and analytical in your decision.

Here is some information regarding the categories available to you
<CATEGORIES INFO>
{categories_info}
</CATEGORIES INFO>

Here is some information explaning the schema of the case information provided to you
<CASE INFORMATION SCHEMA DESCRIPTION>
{tag_info}
</CASE INFORMATION SCHEMA DESCRIPTION>

Here is the actual case information
<CASE INFORMATION>
{case_info}
</CASE INFORMATION>


Thoroughly analyze the case details and categorize them according to the categories provided to you.
"""




# ROSS
path = "/workspace/ROSS_case_list.xlsx"

# Made 2 Manage
# path = "/Users/suryakrishnan/Documents/GitHub/aptean/excel_experiment/Made2Manage_case_list.xlsx"

df = pd.read_excel(path)
# print(len(df))
# print(df.columns)

filtered_df = df.iloc[:1]



class CaseCategorization(BaseModel):
    """This data model captures your response of categorizing the issue according to the instructions in the prompt. Be extremely thoughtful and accurate"""
    reasoning: str = Field(..., description="Provide your reasoning and analysis behind why you are categorizing this issue a particular way.")
    category: Literal['Integration', 'Technical Error', 'Feature Enhancement', 'Module Issue', 'Admin Issue', 'Other']
    new_category_name: Optional[str] = Field(None, description="Only use this if you selected category as other. Provide the category that we should add to capture the class of issues represented by this issue.")

adaptor = PydanticAdaptorOpenRouter(openai_api_key="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")
def classify_case(row):
    case_info = get_case_info(row)

    case_classification_prompt = prompt_template.format(categories_info=categories_info, tag_info=tag_info, case_info=case_info)

    content = [{"type": "text", "text": case_classification_prompt}]
    curr_message = {
        "role": "user",
        "content": content
    }
    message_history = [curr_message]
    case_categorization = adaptor.chat.completions.create(
        pydantic_model=CaseCategorization,
        num_retries=5,
        model="openai/gpt-4o-mini",
        messages=message_history,
        max_tokens=4096,
        stream=False
    )

    case_categorization_dict = case_categorization.model_dump()

    return case_categorization_dict





# classify all cases
case_classification_list = []
with ThreadPoolExecutor(max_workers=42) as executor:
    # Start the load operations and mark each future with its URL
    case_classification_futures = [executor.submit(classify_case, row) for _, row in filtered_df.iterrows()]
    for future in tqdm(as_completed(case_classification_futures), total=len(case_classification_futures)):
        case_classification = future.result()
        case_classification_list.append(case_classification)

case_classification_reasoning_list = []
case_category_name_list = []
case_new_category_name_list = []
for case_classification_dict in case_classification_list:
    reasoning = case_classification_dict['reasoning']
    category = case_classification_dict['category']
    new_category_name = case_classification_dict['new_category_name']
    
    case_classification_reasoning_list.append(reasoning)
    case_category_name_list.append(category)
    case_new_category_name_list.append(new_category_name)

filtered_df.loc['category_reasoning'] = case_classification_reasoning_list
filtered_df.loc['category_name'] = case_category_name_list
filtered_df.loc['new_category_name'] = case_new_category_name_list

filtered_df.to_excel('top_level_classify.xlsx')

# print(len(case_summary_list))

# with open(case_summary_list_filepath, 'w') as f:
#     json.dump(case_summary_list, f, indent=4)

# case_number_list = [row['Case Number'] for _, row in filtered_df.iterrows()]
# with open(case_number_filepath, 'w') as f:
#     json.dump(case_number_list, f, indent=4)
