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

from functools import partial






# ROSS
path = "/workspace/aptean/src/top_level_classify.xlsx"

# Made 2 Manage
# path = "/Users/suryakrishnan/Documents/GitHub/aptean/excel_experiment/Made2Manage_case_list.xlsx"

df = pd.read_excel(path)
# print(len(df))
# print(df.columns)

# category_col_name = "category_name"
# category_vals = ['Integration', 'Technical Error', 'Feature Enhancement', 'Module Issue', 'Admin Issue', 'Other']
# rows_to_extract = df[category_col_name] == 'Module Issue'
# filtered_df = df.loc[rows_to_extract]
filtered_df = df.iloc[:]
print(len(filtered_df))

# component_names = df['Component Name'].dropna().unique()
# new_component_names = []
# for name in component_names:
#     if "Ross" in name:
#         new_name = name.split("Ross ")[1]
#         new_component_names.append(new_name)
#         print(new_name)
#     else:
#         new_component_names.append(name)
#         print(name)

# # print(new_component_names)

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

case_module_extractor_prompt_template = """You are an expert analyst who is tasked with analysing a support case data from Aptean
and figure out which module inside the Aptean product does this support case relate to.

Your task is to accurately identify the module that this support case relates to inside the Aptean
product.

You are given some example module names that are present in similar products to give you an idea of
what are some types of modules that are available in products like these. This module list is not
exhaustive nor is it all applicable to this product. Use it as just an inspiration regarding what counts
as a module in similar products and figure out the module name that this support case belongs to.

<EXAMPLE MODULE NAMES IN SIMILAR PRODUCTS>
- Process Manufacturing
- Accounts Receivable 
- Inventory Control
- Reporting Services
- General Ledger
- Accounts Payables
- Data Collection
- Sales Order Processing
- Data Manager Component
- SCP
- Platform
- Purchase Order Processing
- Applications
- Process Planning
- Maintenance Manager
- Fixed Assets
- System Manager
- Enterprise Viewer
- Project Accounting
- Trade Promotions
- EDI
- EMF Base Component
- Customizations
- Advanced Reporting
- EMF
- Materials Management
- Sales Analysis
- EMF Processes
- F&B (cWMS Integration)
- General Business Kit
- Reporting Services Reports
- Certified Extension
</EXAMPLE MODULE NAMES IN SIMILAR PRODUCTS>

Here is some information explaning the schema of the case information provided to you
<CASE INFORMATION SCHEMA DESCRIPTION>
{tag_info}
</CASE INFORMATION SCHEMA DESCRIPTION>

Here is the actual case information
<CASE INFORMATION>
{case_info}
</CASE INFORMATION>

Thoroughly analyse the case and figure out the module name that this support case belongs to
"""



openai_client = OpenAI()
adaptor = PydanticAdaptorOpenRouter(openai_client=openai_client)

class ModuleExtractor(BaseModel):
    """This data model captures your response of extracting the module that this support case belongs to"""
    reasoning: str = Field(..., description="Provide your reasoning and analysis behind your conclusion on the module this support case belongs to.")
    module_name: str = Field(..., description="Name of the module this support case belongs to")

def get_case_module(row):
    case_info = get_case_info(row)
    case_module_extraction_prompt = case_module_extractor_prompt_template.format(tag_info=tag_info, case_info=case_info)

    content = [{"type": "text", "text": case_module_extraction_prompt}]
    curr_message = {
        "role": "user",
        "content": content
    }
    message_history = [curr_message]
    case_module_extractor = adaptor.chat.completions.create(
        pydantic_model=ModuleExtractor,
        num_retries=5,
        # model="gpt-4o-mini",
        model="gpt-4o-mini",
        messages=message_history,
        temperature=0,
        max_tokens=4096,
        stream=False
    )

    case_module_extractor_dict = case_module_extractor.model_dump()

    return case_module_extractor_dict


def is_same_module(first_module_name, second_module_name):
    same_module_name_analysis_prompt = same_module_name_analysis_prompt_template.format(first_module_name=first_module_name, second_module_name=second_module_name)

    content = [{"type": "text", "text": same_module_name_analysis_prompt}]
    curr_message = {
        "role": "user",
        "content": content
    }
    message_history = [curr_message]
    merge_module_result = adaptor.chat.completions.create(
        pydantic_model=SameModuleNameAnalysis,
        num_retries=5,
        # model="gpt-4o-mini",
        model="gpt-4o-mini",
        messages=message_history,
        temperature=0,
        max_tokens=4096,
        stream=False
    )

    merge_module_result_dict = merge_module_result.model_dump()

    return merge_module_result_dict


# count = 0
# for _, row in filtered_df.iterrows():
#     if count >= 5:
#         break
#     print(row['Component Name'])
#     case_module_extractor_dict = get_case_module(row)
#     print(case_module_extractor_dict['module_name'])
#     print("\n\n")
#     count += 1
# exit()



# # classify all cases
# case_module_extraction_list = []
# with ThreadPoolExecutor(max_workers=42) as executor:
#     # Start the load operations and mark each future with its URL
#     case_module_extraction_futures_list = [executor.submit(get_case_module, row) for _, row in filtered_df.iterrows()]
#     for future in tqdm(as_completed(case_module_extraction_futures_list), total=len(case_module_extraction_futures_list)):
#         case_module_extraction = future.result()
#         case_module_extraction_list.append(case_module_extraction)

# case_module_reasoning_list = []
# case_module_name_list = []
# for case_classification_dict in case_module_extraction_list:
#     reasoning = case_classification_dict['reasoning']
#     module_name = case_classification_dict['module_name']
    
#     case_module_reasoning_list.append(reasoning)
#     case_module_name_list.append(module_name)

# filtered_df.loc[:, 'module_name_reasoning'] = case_module_reasoning_list
# filtered_df.loc[:, 'module_name'] = case_module_name_list

# filtered_df.to_excel('extract_module.xlsx')

# print(filtered_df["module_name"].value_counts())


# deduplicate module names 


same_module_name_analysis_prompt_template = """You are given two module names referring to a module in a ERP product 
written by two different people. Sometimes, the different names refer to the same module in the ERP product 
and sometimes they refer to different modules in the ERP product. Sometimes, the same module name are just written 
slightly differently by different people based on their understanding or memory, but semantically points to the same
module in the ERP product.

Your task is to analyse and answer whether the two given module names refere to the same module or whether they
refer to different modules in this ERP product.

<MODULE NAME 1>
{first_module_name}
</MODULE NAME 1>

<MODULE NAME 2>
{second_module_name}
</MODULE NAME 2>

Thoroughly analyse the module names and answer whether to refere the same module name in the ERP or not.
"""
class SameModuleNameAnalysis(BaseModel):
    """This is the data model that captures the result of your analysis on whether the given module names
    refer to the same module in the ERP product or not"""
    reasoning: str = Field(..., description="Your analysis/reasoning on whether the given module names refere to the same module or not")
    is_same_module: bool = Field(..., description="Your conclusion on wheter the given module names refer to the same module or not")

comebine_module_name_prompt_template = """You are given a set of module names that all refer to the same module in a 
ERP product. These names are just how different people refer to the same module in the ERP product based on their memory / understanding
of the product. 

Your task is to come up with a single definitive name to refer the this module. Use the given module names that different people use
as a guidance to understand what the module is and what types of names they like using and come up with a name that will be shown
to everyone as definitely the way to refer to this module of the ERP product going forward

<MODULE NAME LIST>
{module_name_list}
</MODULE NAME LIST>

Thoroughly analyse the module names and come up with a common and definitive name to refer to this module in the ERP product going forward
"""
class CombineModuleName(BaseModel):
    """This is the data module that captures the result of your analysis on the module names and your answer to the single definitve name
    to refer to the given module in the ERP product"""
    reasoning: str = Field(..., description="Your analysis/reasoning on what the single definitive name for this module should be")
    module_name: str = Field(..., description="Your answer on what the single definitive name for this module should be.")



df = pd.read_excel('extract_module.xlsx')
module_name_list = df['module_name'].unique()

# with open('temp.json', 'w') as f:
#     json.dump(module_name_list.tolist(), f, indent=4)
# exit()

merged_module_name_dict = {}
while(len(module_name_list) > 0):
    query_module_name = module_name_list[0]
    same_module_name_list = [query_module_name]
    print(query_module_name)
    if len(module_name_list) > 1:
        partial_is_same_module = partial(is_same_module, first_module_name=query_module_name)
        with ThreadPoolExecutor(max_workers=42) as executor:
            is_same_module_futures_dict = {}
            for curr_module_name in module_name_list[1:]:
                future = executor.submit(partial_is_same_module, second_module_name=curr_module_name)
                is_same_module_futures_dict[future] = curr_module_name

            for future in tqdm(as_completed(is_same_module_futures_dict), total=len(is_same_module_futures_dict)):
                is_same_module_result_dict = future.result()
                if is_same_module_result_dict['is_same_module']:
                    curr_module_name = is_same_module_futures_dict[future]
                    same_module_name_list.append(curr_module_name)

                    # print(is_same_module_result_dict)
                    # print(curr_module_name)
                    # print("\n")
                
    print(same_module_name_list)
    module_name_list = [name for name in module_name_list if name not in same_module_name_list]

    if len(module_name_list) == 1:
        combine_module_name = module_name_list[0]
    else:
        combine_module_name_prompt = comebine_module_name_prompt_template.format(module_name_list=same_module_name_list)
        content = [{"type": "text", "text": combine_module_name_prompt}]
        curr_message = {
            "role": "user",
            "content": content
        }
        message_history = [curr_message]
        combine_module_name_result = adaptor.chat.completions.create(
            pydantic_model=CombineModuleName,
            num_retries=5,
            # model="gpt-4o-mini",
            model="gpt-4o-mini",
            messages=message_history,
            temperature=0,
            max_tokens=4096,
            stream=False
        )
        combine_module_name = combine_module_name_result.module_name
        print(combine_module_name)

    for module_name in same_module_name_list:
        merged_module_name_dict[module_name] = combine_module_name


merged_module_name_list = []
for _, row in df.iterrows():
    module_name = row["module_name"]
    merged_module_name = merged_module_name_dict[module_name]
    merged_module_name_list.append(merged_module_name)


df.loc[:, 'merged_module_name'] = merged_module_name_list
df.to_excel('merged_module_name.xlsx')

print(df['merged_module_name'].value_counts())


# use the response + missing to do it
    # keep doing this until done
