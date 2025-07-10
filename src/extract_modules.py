from typing import List, Union
import json
from tqdm import tqdm

import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from pydantic import BaseModel, Field
from openai import OpenAI
from PydanticAdaptorAnthropic import PydanticAdaptorAnthropic
from PydanticAdaptorOpenRouter import PydanticAdaptorOpenRouter

from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
adaptor = PydanticAdaptorOpenRouter(openai_cient=openai_client)


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


initial_module_names_list = [
    "Process Manufacturing",
    "General Ledger",
    "Purchase Order Processing",
    "Inventory Control",
    "Data Manager Component",
    "Process Planning",
    "Customizations",
    "Accounts Receivable",
    "Applications",
    "Accounts Payables",
    "Sales Order Processing",
    "SCP",
    "Reporting Services",
    "Materials Management",
    "Supply Chain Planning (SCP)",
    "Resource Planning Management (RPM)",
    "Maintenance Manager",
    "Fixed Assets",
    "Accounts Payable",
    "System Manager",
    "Credential Management",
    "Data Collection",
    "User Management",
    "Project Accounting",
    "Enterprise Viewer",
    "Quality Control Management",
    "Trade Promotions",
    "Recipe Management",
    "Quality Management",
    "Cost Management",
    "Platform",
    "Quality Control",
    "Advanced Planning and Scheduling (APS)",
    "Security Manager",
    "TraceExpress",
    "Trace Express",
    "Reporting Services Reports",
    "EDI",
    "SCP (Supply Chain Planning)",
    "Security Management",
    "Product Costing",
    "Email Integration",
    "Accounts Payable / Accounts Receivable",
    "Security and Auditing",
    "Campaign Management",
    "Product Master",
    "Manufacturing Inquiries",
    "Document Management",
    "EMF Processes",
    "EMF",
    "Database Management",
    "Shipping Management",
    "Advanced Reporting",
    "Shipping and Order Processing",
    "ERP",
    "Mobile Applications",
    "Product Master Maintenance",
    "Web Service Module",
    "Sales Analysis",
    "User Setup",
    "License Management",
    "Printing Services",
    "Ross Mobile Browser",
    "Contract Management",
    "Advanced Planning and Scheduling",
    "User Interface",
    "User Account Management",
    "Intercompany Subledger",
    "ERP Services",
    "Facilities Management",
    "Tabware",
    "General Business Kit",
    "Customer Management",
    "Job Management",
    "User Accounts Management",
    "Ross",
    "Batch Processing",
    "Costing",
    "Manufacturing",
    "Job Costing",
    "IAF",
    "Services",
    "User Configuration Management",
    "Product Maintenance",
    "MRP",
    "Label Management",
    "IAF Manager",
    "Product Lifecycle Management",
    "Configuration Management",
    "Ross ERP",
    "Product Inquiry",
    "ERP Administration",
    "Company Controls",
    "Currency Management",
    "Accounts Receivable / Accounts Payable",
    "MRP (Material Requirements Planning)",
    "Process Control",
    "F&B (cWMS Integration)",
    "Software Compatibility",
    "User Security Management",
    "Licensing Management",
    "Shipping Order Processing",
    "User Authentication",
    "Demand Planning",
    "Customer Service",
    "Label Printing",
    "User Access Management",
    "User Profile",
    "Web Services",
    "Event Management Framework (EMF)",
    "Content Management",
    "Pyramid Module",
    "Cash Management",
    "Print Services",
    "Quality Assurance / Testing",
    "Accounts Maintenance",
    "RPM",
    "Mobile Transaction Processing",
    "Aptean Connect OATKA",
    "Batch Manager",
    "Data Warehouse",
    "Traceability",
    "Financials",
    "Cost Accounting",
    "Print Management",
    "Software License Management",
    "User Access",
    "Scheduling",
    "Aptean Connect",
    "Accounts",
    "Warehouse Management",
    "BPM",
    "Webdesktop Configuration",
    "Product Support Life",
    "Configuration Tool",
    "Process Accounting",
    "Product Allocation",
    "WebDesktop",
    "Customer Master",
    "Localization",
    "ROSS",
    "Shipping and Logistics",
    "Sales Forecasting",
    "Text Formatter",
    "Certified Extension",
    "Shipping and Logistics Management",
    "User Administration",
    "Shipping",
    "Data Management",
    "Mobile IAF",
    "Financial Security Management",
    "Shipping Operations",
    "Supply Chain Planning",
    "Product Management",
    "Company Setup",
    "Accounts Management",
    "Shipping and Invoicing",
    "Process Management",
    "Inventory Control, Process Manufacturing, Purchase Order Processing, Sales Order Processing",
    "Integration Services",
    "Warehouse Transfer Receipt",
    "Transportation Management",
    "Aptean Ross",
    "Cloud Printing",
    "Labor Management",
    "SaaS",
    "RUN EXECUTABLE",
    "Security Controls",
    "Aptean Cloud Reporting",
    "Metadata Management",
    "User Security",
    "Case Management",
    "Cloud Printing Services",
    "Supplier Management",
    "e-Signature Management",
    "Web Services Module",
    "Accounts Payables and Accounts Receivables",
    "Job Accounting",
    "User Setup and Security Manager",
    "Printing and Label Management",
    "Production Planning",
    "Metadata Manager",
    "Warehouse Management System",
    "IAF File Manager",
    "Job Inquiry",
    "Routing & Scheduling",
    "Shipping and Order Management",
    "Purchasing",
    "Self-Service",
    "Licensing",
    "Job Scheduling",
    "Aptean Analytics",
    "Account Security",
    "Mobile/Modern",
    "Webdesktop",
    "Action Center",
    "Support Services",
    "Accounts Payables / Accounts Receivable",
    "Planning",
    "Shipping and Distribution",
    "Weigh & Dispense",
    "IAF Management",
    "Job Control",
    "Shipping and Dispatch Management",
    "Lock Manager",
    "Reconciliation Module",
    "Despatch Management",
    "User Access and Security Manager",
    "Transfer Order Processing",
    "Weigh and Dispense",
    "Shipping and Inventory Management",
    "Cloud Reporting",
    "Lock Management",
    "User Access Setup",
    "Advanced Pricing",
    "User Setup and Security",
    "Product Master Management",
    "Accounts Payable and Accounts Receivable",
    "Disaster Recovery",
    "Facility Management",
    "Banking Module",
    "User Authentication Management",
    "Graphical Scheduling",
    "Distribution Center (DC) Module",
    "CASH",
    "Job Reporting",
    "User Personalization",
    "BPO",
    "Pharmacy License Management",
    "Job Processing"
]

same_module_name_analysis_prompt_template = """You are given a set of module names referring to different modules in a ERP product 
Sometimes, the multiple names in the given list refer to the same module in the ERP product and sometimes they refer to different modules in the ERP product. 
Sometimes, the same module name are just written slightly differently by different people based on their understanding or memory, but semantically points to the same
module in the ERP product.

Your task is to analyse all the module names thoroughly and deduplicate the names that all refer to the same ERP module and group the sets of
names by the different ERP modules

Here is all the module names 
<MODULE NAMES>
{module_name_list}
</MODULE NAMES>

Instructions
- Make sure to not ignore any name. Every single name should be accounted for.
- If a name is unique and is uniquely referring to a module. Add that to a module grouping with it as the only name and use that name itself
for the combined name.
- Every name in the given <MODULE NAMES> should be put in a group.
- Each name in the given <MODULE NAMES> can only be put in a group once.
- Make sure to keep the name exactly the same including upper case and lower case.

Thoroughly analyse the module names and group the names in sets of ones that all refere to the same ERP module.
"""

class ModuleGrouping(BaseModel):
    """This data model captures information on all the names that all refere to the same ERP module and a definitive
    name that should be used to refer to this ERP module by everyone going forward
    
    Instructions
    - Make sure to not ignore any name. Every single name should be accounted for.
    - If a name is unique and is uniquely referring to a module. Add that to a module grouping with it as the only name and use that name itself
    for the combined name.
    - Make sure to keep the name exactly the same including upper case and lower case.
            
    """
    reasoning: str = Field(..., description="Reasoning behind grouping together all the names that all refer to the same ERP module and your decision on figuring out a common and definitive name to refer to this module going forward.")
    name_list: List[str] = Field(..., description="All the names that all refer to the same particular ERP module")
    combined_name: str = Field(..., description="A definitive name for this ERP module that will be used by everyone going forward")

class ModuleNameClustering(BaseModel):
    """This is the data model that captures the result of your analysis and result on deduplicating all the names that refer to the
    same ERP module and grouping them by the different ERP modules"""
    overall_analysis: str = Field(..., description="Your overall analysis of this task and the reasoning behind your approach and solution")
    module_groups: List[ModuleGrouping] = Field(..., description="Set of all the unique ERP modules referenced here and information all the names that all refer to the same module.")

# Add missing names
group_missing_names_prompt_template = """You are given a set of module names referring to different modules in a ERP product 
Sometimes, the multiple names in the given list refer to the same module in the ERP product and sometimes they refer to different modules in the ERP product. 
Sometimes, the same module name are just written slightly differently by different people based on their understanding or memory, but semantically points to the same
module in the ERP product.

You are also given a set of definitive group names referring to different modules in the ERP product. If any of the module names refer to the
same module as the one represented by one of the group names, simply conclude that this module name belongs to the group defined by that group
name. If a set of module names or a single module name is not represented by any of the given group names, conclude that new group should be created
and add the corresponding module names to this new group

Here are all the group names that you can associate modules to if they semnatically belong to this group.
<MODULE GROUP NAMES>
{module_group_names}
</MODULE GROUP NAMES>


Here is all the module names 
<MODULE NAMES>
{module_name_list}
</MODULE NAMES>

Instructions
- Make sure to not ignore any name. Every single name should be accounted for.
- If a name is unique and is uniquely referring to a module. Add that to a module grouping with it as the only name and use that name itself
for the combined name.
- Every name in the given <MODULE NAMES> should be put in a group.
- Each name in the given <MODULE NAMES> can only be put in a group once.
- Make sure to keep the name exactly the same including upper case and lower case.

Thoroughly analyse the module names and group the names in sets of ones that all refere to the same ERP module.
"""

#TODO needs to be a literal of existing group names
class AddToExistingGrouping(BaseModel):
    """This is the data model that captures the decision to add one or a set of module names to one of the existing groups provided to you."""
    reasoning: str = Field(..., description="analysis/reasoning behind your decision to associate the module name with this group name")
    name_list: List[str] = Field(..., description="List of module names to associate with this module group name")
    group_name: str = Field(..., description="The provided module group name that we are associating these module names to")

class ModuleNameGroupAllocation(BaseModel):
    """This is the data model that captures the result of your analysis and result on deduplicating all the module names that refer to the
    same ERP module and grouping them by the different ERP modules"""
    overall_analysis: str = Field(..., description="Your overall analysis of this task and the reasoning behind your approach and solution")
    module_groups: List[Union[ModuleGrouping, AddToExistingGrouping]] = Field(..., description="Set of all the unique ERP modules referenced here and information all the names that all refer to the same module.")

def cluster_initial_module_names(module_names_list) -> ModuleNameClustering:
    same_module_name_analysis_prompt = same_module_name_analysis_prompt_template.format(module_name_list=module_names_list)
    content = [{"type": "text", "text": same_module_name_analysis_prompt}]
    curr_message = {
        "role": "user",
        "content": content
    }
    message_history = [curr_message]

    raw_module_grouping_response = openai_client.chat.completions.create(
        model="o1",
        messages=message_history,
        # max_tokens=4096,
        stream=False
    )

    raw_model_grouping_text = raw_module_grouping_response.choices[0].message.content
    
    #TODO remove this just for debugging
    with open("raw_model_grouping_text.md", "w") as f:
        f.write(raw_model_grouping_text)

    assistant_message = {
        "role": "assistant",
        "content": raw_model_grouping_text
    }

    format_prompt = f"""format the extracted grouping data in the given data model. Be extremely thorough and accurate.
    Here is the content to format
    """
    content = [{"type": "text", "text": format_prompt}]
    curr_message = {
        "role": "user",
        "content": content
    }
    message_history.append(assistant_message)
    message_history.append(curr_message)
    module_name_clustering = adaptor.chat.completions.create(
        pydantic_model=ModuleNameClustering,
        num_retries=1,
        # model="gpt-4o-mini",
        model="o1",
        messages=message_history,
        # max_tokens=4096,
        stream=False
    )

    return module_name_clustering


def add_missing_module_groups(missing_module_names_list, existing_module_groups_list) -> ModuleNameGroupAllocation:
    group_missing_names_prompt = group_missing_names_prompt_template.format(module_group_names=existing_module_groups_list, module_name_list=missing_module_names_list)
    
    content = [{"type": "text", "text": group_missing_names_prompt}]
    curr_message = {
        "role": "user",
        "content": content
    }
    message_history = [curr_message]

    module_name_group_allocation_model = adaptor.chat.completions.create(
        pydantic_model=ModuleNameGroupAllocation,
        num_retries=1,
        # model="gpt-4o-mini",
        model="o1",
        messages=message_history,
        # max_tokens=4096,
        stream=False
    )

    return module_name_group_allocation_model


def get_missing_module_names_list(original_names_list, module_group2module_names):

    module_names2module_group = get_module_names2module_group(module_group2module_names)
    missing_module_names_list = []
    for original_name in original_names_list:
        if original_name not in module_names2module_group:
            missing_module_names_list.append(original_name)
    
    return missing_module_names_list


#TODO check if there are duplicate module names, makes up module names, missing module names
def get_module_group2module_names(module_name_clustering: ModuleNameClustering):
    module_group2module_names = {}
    for module_group in module_name_clustering.module_groups:
        group_name = module_group.combined_name
        group_module_name_list = module_group.name_list
        
        module_group2module_names[group_name] = group_module_name_list
    
    return module_group2module_names


#TODO error checking on one module name being assigned to multiple groups
def get_module_names2module_group(module_group2module_names):
    module_names2module_group = {}
    for module_group, group_module_name_list in module_group2module_names.items():
        if group_module_name_list is None:
            print(module_group, group_module_name_list)
        for module_name in group_module_name_list:
            module_names2module_group[module_name] = module_group
    return module_names2module_group


def extend_module_group2module_names(module_group2module_names, module_name_group_allocation_model: ModuleNameGroupAllocation):
    for module_group in module_name_group_allocation_model.module_groups:
        if isinstance(module_group, ModuleGrouping):
            group_name = module_group.combined_name
            group_module_name_list = module_group.name_list
            
            if group_module_name_list is None:
                print(group_name, group_module_name_list)
            #TODO catch this situation properlyt and fix it. should never happen.
            if group_name in module_group2module_names:
                module_group2module_names[group_name] = module_group2module_names[group_name].extend(group_module_name_list)
            else:
                module_group2module_names[group_name] = group_module_name_list
        else:
            group_name = module_group.group_name
            group_module_name_list = module_group.name_list

            if group_module_name_list is None:
                print(group_module_name_list)

            if group_name not in module_group_names_list:
                print('THis is not a valid group name', group_name, group_module_name_list)
                pass
            else:
                module_group2module_names[group_name] = module_group2module_names[group_name].extend(group_module_name_list)
    
    return module_group2module_names







if __name__ == "__main__":
    # INPUT -> top level classify excel
    # OUTPUT -> Module name for everything excel

    # extract all module names
    # deduplicate module names


    # ROSS
    path = "/workspace/aptean/src/top_level_classify.xlsx"

    # Made 2 Manage
    # path = "/Users/suryakrishnan/Documents/GitHub/aptean/excel_experiment/Made2Manage_case_list.xlsx"

    df = pd.read_excel(path)


    # category_col_name = "category_name"
    # category_vals = ['Integration', 'Technical Error', 'Feature Enhancement', 'Module Issue', 'Admin Issue', 'Other']
    # rows_to_extract = df[category_col_name] == 'Module Issue'
    # filtered_df = df.loc[rows_to_extract]
    filtered_df = df.iloc[:]
    print(len(filtered_df))



    # extract module name for alll cases
    case_module_extraction_list = []
    with ThreadPoolExecutor(max_workers=42) as executor:
        # Start the load operations and mark each future with its URL
        case_module_extraction_futures_list = [executor.submit(get_case_module, row) for _, row in filtered_df.iterrows()]
        for future in tqdm(as_completed(case_module_extraction_futures_list), total=len(case_module_extraction_futures_list)):
            case_module_extraction = future.result()
            case_module_extraction_list.append(case_module_extraction)

    case_module_reasoning_list = []
    case_module_name_list = []
    for case_classification_dict in case_module_extraction_list:
        reasoning = case_classification_dict['reasoning']
        module_name = case_classification_dict['module_name']
        
        case_module_reasoning_list.append(reasoning)
        case_module_name_list.append(module_name)

    filtered_df.loc[:, 'module_name_reasoning'] = case_module_reasoning_list
    filtered_df.loc[:, 'module_name'] = case_module_name_list

    filtered_df.to_excel('extract_module.xlsx')


    # deduplicate module names
    module_name_clustering_model = cluster_initial_module_names(initial_module_names_list)
    module_group2module_names = get_module_group2module_names(module_name_clustering_model)
    missing_module_names_list = get_missing_module_names_list(initial_module_names_list, module_group2module_names)
    print(f"missing_module_names_list - {missing_module_names_list}")
    print(f"len missing module name - {len(missing_module_names_list)}")
    print("-" * 25)
    while len(missing_module_names_list) > 0:
        module_group_names_list = list(module_group2module_names.keys())
        module_name_group_allocation_model = add_missing_module_groups(missing_module_names_list, module_group_names_list)
        
        module_group2module_names = extend_module_group2module_names(module_group2module_names, module_name_group_allocation_model)
        missing_module_names_list = get_missing_module_names_list(initial_module_names_list, module_group2module_names)

        print(f"missing_module_names_list - {missing_module_names_list}")
        print(f"len missing module name - {len(missing_module_names_list)}")
        print("-" * 25)


    with open('module_group2module_names.json', 'w') as f:
        json.dump(module_group2module_names, f, indent=4)