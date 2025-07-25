import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Literal, List
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from PydanticAdaptorOpenRouter import PydanticAdaptorOpenRouter
from knowledgebase_analysis import kb_analysis
from summary_report_generate import report_process

tag_info = """Here is some context on the different pieces of information you may be given
access to as part of analysing the case. Each of the below menitoned tags may or may not be
present for each case

<TAG INFO>
1. SUBJECT - This is the initial title or summary provided by the **customer** when raising the case. It usually offers a brief indication of the reported issue and helps establish the **high-level context** of the case.
2. DESCRIPTION - This represents the description of the case and some notes around it, written by the internal support team while discussing the case amongst themselves as they work to resolve it. These notes are typically internal-facing and may contain technical jargon or references to known issues or configurations.
3. POSTS - This represents ad-hoc, often unstructured information present in internal logs, chat notes, or tracking documents as the support team investigates the issue. These entries may contain useful insights, but also a fair amount of noise, such as meeting schedules, team coordination details, or status updates that are not directly relevant to the technical root cause.
4. EMAILS - This section contains the chronological conversation between the **support agent(s)** and the **customer**. It reflects the full communication trail starting from the customer's first report of the issue through to resolution or closure.
   These communications are often informal, and may include misspellings, mixed technical and non-technical language, and emotional tone (e.g., customer frustration or urgency). They are invaluable for understanding the **true impact of the issue**, the **steps taken**, and **what actually resolved it**.
</TAG INFO>
"""

case_prompt_template = """You are a support analyst at Aptean reviewing a subset of technical support cases that have been grouped together 
due to their similar issue patterns and underlying themes. Your task is to analyze the commonalities across these cases and propose a clear, 
descriptive topic that accurately summarizes the core problem or theme represented by this cluster. Provide a detailed explanation of why this 
topic best represents the pattern observed in these cases. Do not assume anything and your analysis should be purley on the basis of the data provided.

Here is some information regarding the categories available to you and Return a consistent category name.
<CATEGORIES INFO>
{categories_info}
</CATEGORIES INFO>

For each group of support case groups:

- Describe the actual root cause and recommendation that needs to be implemented to solve all the support case groups in this group and prevent them from recurring in the future.  
Be extremely thorough, clear, and complete in your explanation.

Use the following guidelines based on the category assigned:

- If the category is `Feature Enhancement`:  
  *Describe the feature that needs to be added in detail.*  
  Explain what functionality is missing, how it should behave, which users it affects, and how it will solve the cases. Include UI elements, workflow behavior, configuration options, or system logic that needs to be introduced or improved.

- If the category is `Software Bug`:  
  *Describe in detail what bug needs to be fixed, and if it's clear how, mention that as well.*  
  Identify where the faulty behavior occurs, what is causing it, and what the expected behavior is. If known, provide insight into the specific logic, condition, or module that needs to be corrected.

- If the category is `Admin Issue`:  
  *Describe in detail what administrative task needs to be automated, and how it can be automated efficiently.*  
  Identify the repetitive or error-prone task (e.g., user setup, permissions, config), and propose automation through templates, UI flows, batch tools, or validation logic.

- If the category is `Integration`:  
  Identify the failing or unreliable integration (e.g., API, sync, file exchange). Explain the failure cause (e.g., schema mismatch, timeout, missing validation), and recommend specific integration fixes like retries, better error handling, monitoring, or documentation improvements.

- If the category is `Module Issue`:  
  Describe the specific module behavior that is inconsistent, incomplete, or breaking. Explain what functional logic or configuration needs to change to make the module reliable and meet the intended use cases.

- If the category is `Data Error`:  
  Explain how the system is producing or accepting invalid data. This could be due to weak validations, poor imports, sync mismatches, or missing constraints. Recommend data checks, repair routines, validation improvements, or prevention steps.

- If the category is `Setup Issue`:  
  Identify how users are misconfiguring the system, and why. Recommend improved guidance, smarter defaults, mandatory fields, setup validation flows, or step-by-step onboarding to help users complete the configuration correctly.

- If the category is `Process Gap`:  
  Describe the business process weakness, such as missing approvals, undefined ownership, or incomplete workflows. Recommend changes like enforcing process steps, user assignments, business rule validations, or escalation flows.

- If the category is `Educational Issue`:  
  Identify what users are not understanding, misusing, or unaware of. Recommend training content, how-to guides, in-product tours, videos, or contextual help to improve understanding.

- If the category is `Other`:  
  Use analytical judgment to identify the root cause and solution. Provide a strong rationale for the issue and recommend a preventive approach, such as cross-team process realignment, new tooling, or specialized intervention.

Be structured, technical, and specific in both your root cause and your recommendation. Avoid vague terms like "issue", "problem", or "error". Write in a way that would inform product, support, or engineering teams about what to actually fix or implement.

You are expected to produce a structured response containing the following fields exactly as specified. Do not omit or rename any field:
- lv1
- lv2
- lv3
- insight_category
- customer_query
- case_summary
- case_root_cause
- case_recommendation
- kb_candidate (must be either 'Y' or 'N' based on strict criteria)

Here is some information explaning the schema of the case information provided to you
<CASE INFORMATION SCHEMA DESCRIPTION>
{tag_info}
</CASE INFORMATION SCHEMA DESCRIPTION>>

Here is the actual case information
<CASE INFORMATION>
{case_info}
</CASE INFORMATION>

"""

categories_info = """
    Here are the error categories: 

    - Integration: If an issue involves either an integration feature - whether its a bug, a general issue,
    feature or any other type of issue, classify it as an integration

    - Software Bug: If an issue directly involves a technical bug and is not related to integrations, then 
    classify it as a technical error. Some examples of technical bugs are things like things freezing, things being too slow, 
    features not working as intended, etc.

    - Feature Enhancement: If an issue is directly related to a module or a set of modules and involves the addition of a
    new feature to the product, then classify it as a feature enhancement.

    - Module Issue: If an issue is directly related to a module or a set of modules and is not related to integrations, 
    a technical bug, or a feature enhancement, then classify it as a module issue. 

    - Admin Issue: If an issue involves admin related tasks such as adding users, removing users, managing user authentication 
    (including login issues, password resets, account lockouts), configuring user roles and permissions, IP whitelisting, or any other 
    administrative or user management related task, then classify it as an Admin Issue.

    - Data Error: If an issue is directly related to data corruption issue or any form of data issue but is not related to any integration issue, 
    feature enhancement, any module specific issue or any other technical issue then classify it as a Data Error.

    - Setup Issue: If an issue is directly related to a system setup issue, or a product configuration issue but is not related to any integration issue, data corruption issue, feature enhancement, 
    any module specific issue or any other technical issue then classify it as a Setup Issue.

    - Process Gaps: If an issue is directly not related to any issue but is more like a process gap such as Delays in responding or resolving issues, 
    Incorrect team or priority assignment, Premature case closure or repeat reopenings, Customers needing to follow up multiple times, 
    Support policies or SLAs not followed properly etc, then classify it as a Process Gap

    - Educational Issue: If an issue is directly not related to any issues, process gaps in support process or cannot be addressed by a simple knowledge base article, 
    and a tutorial that is properly educational could be better suited for the given issue or situation, then classify it as an educational issue.

    - Other Miscellaneous: If an issue does not fall under any of these categories then classify it under Other Miscellaneous.
    Only do this if the issue cannot be clearly placed in any of the above mentioned categories.
"""

adaptor = PydanticAdaptorOpenRouter(openai_client=None, openai_api_key=os.getenv('OPENROUTER_API_KEY'))

class CaseClassification(BaseModel):
    lv1: str = Field(..., description="Recommend a group name that represents broad functional area such as Access, Reporting, Integration, Admin, Very high level module etc. There should not be duplicates or similar sounding group names (Example: 'Performance' and 'Performance Issues' are same and only one of the two should exist) within this group and should not be duplicates or similar sounding names across  lv2 and lv3 groups. Also, the names should not match any of the following names 'Integration', 'Software Bug', 'Feature Enhancement', 'Module Issue', 'Admin Issue', 'Data Error', 'Setup Issue', 'Process Gap', 'Educational Issue'). Come up with an alternate relevant name if it matches any of these names, Use consistent Title Case and ensure the name is a **singular category** (e.g., use 'Report' instead of 'Reports')")
 
    lv2: str = Field(..., description="Recommend a group name that represents specific product modules, workflow, or component within the product that would typically come under their parent lv1 category. There should not be duplicates or similar sounding names within this group and should not be duplicates similar sounding names across  lv1 and lv3 groups. Note that lv2 will have their respective lv3 children")
 
    lv3: str = Field(..., description="Recommend a group name that represents concrete technical symptoms or failure pattern (e.g., 'Login screen hangs', 'Incorrect totals in report', 'Specific Login Failures', 'Specific Product modules', 'Specific Data Discrepancies', 'Specific Setup issues', 'Specific Errors',  ). There should not be duplicates or similar sounding names within this group and should not be duplicates or similar sounding names across  lv1 and lv2 groups. If you come across suplicates or similar sounding names then try to be more specific in naming them so that users can identify the differences through the name. Note that lv3 names are the most granular leaf level categories that have their respective lv2 parents")

    insight_category: Literal['Integration', 'Software Bug', 'Feature Enhancement', 'Module Issue', 'Admin Issue', 'Data Error', 'Setup Issue', 'Process Gap', 'Educational Issue', 'Other'] = Field(
        ..., 
        description="Your task is to categorize this issue according to the categories provided to you. Only select one category per case and be extremely thoughtful and analytical in your decision."
    )

    customer_query: str = Field(
        ..., 
        description="Describe in details from the customer's perspective what their queries were by summarising what all those customers had in common while narrating the issues they faced while raising these support tickets. Do not make it specific to one customer but generalize this across different customers that raised these similar issues."
    )

    case_summary: str = Field(
    ..., 
    description="Provide a detailed and neutral summary of the issue derived from the grouped support cases. Include: (1) what the core problem or failure was across the cases, (2) what investigative or troubleshooting steps were taken by support or engineering teams, and (3) what the final resolution, fix, or workaround was. This should be fact-based and capture what actually happened during resolution."
    )

    case_root_cause: str = Field(
        ..., 
        description="Clearly explain the root cause of the issue across the analyzed cases. Your explanation should be aligned with one of the following categories: Integration, Software Bug, Feature Enhancement, Module Issue, Admin Issue, Data Error, Setup Issue, Process Gap, Educational Issue or Other. Be specific about what failed technically or procedurally."
    )
    case_recommendation: str = Field(
        ..., 
        description="Based on the root cause and the insight category, provide very detailed actionable recommendations as to what exactly should be done to address all such issues. These may include one of the following- fixing a software bug, proposing a feature change or enhancement, fixing a specific data issue, addressing an integration issue,  preventing a potential configuration or setup issue , closing a process gap, automating a task, adding Knowledgebase content, or initiating training. Your recommendation should succinct and should clearly correspond to the category."
    )

    kb_candidate: Literal['Y', 'N'] = Field(
        ...,
        description=(
            "Determine if the cases could be addressed by a Knowledgebase (KB) article with absolute certainty that can enable users to self serve so they could address such issues themselves without taking support professionals help."
            "You must assign one of the two values:"
            "kb_candidate: Literal['Y', 'N']"
            "Following are the criteria- Section 'A' for 'Y' and Section 'B' for 'N' "

            "A. Mark 'Y' only if ALL of the following 'A.1', 'A.2', 'A.3', 'A.4' are true:"
            "A.1. The issue occurred across multiple customers — not a one-off or unique to a specific environment."
            "A.2. The root cause and resolution are generalizable — not tied to customer-specific configurations, data, or scripts."
            "A.3. The fix or workaround is clear, repeatable, and can be executed by support staff or end users (e.g., configuration steps, usage clarification)."
            "A.4. The solution can be safely documented and reused in future similar cases."

            "B. Otherwise, mark 'N' if ANY of the following are true:"
            "B.1. The issue is unique to a single customer, environment, or custom setup."
            "B.2. It requires engineering intervention such as a code fix, patch, or product enhancement."
            "B.3. It involves third-party dependencies, sensitive backend changes, or one-off data cleanup."
            "B.4. The workaround is complex, not clearly actionable, or not reusable across other cases."
            "B.5. The case is related to a **feature enhancement**, a **software or code bug/fix**, or a **data error** that requires backend correction and cannot be addressed through documentation alone."
            "Be conservative: only mark 'Y' when the issue is recurring, self-resolvable, and the resolution is well-suited for documentation."
        )
    )

class CategoryGroup(BaseModel):
    category_name: str = Field(..., description="The name of the inferred high-level category.")
    cluster_ids: List[int] = Field(..., description="List of unique cluster IDs belonging to this category.")

class ClusterGroupingResponse(BaseModel):
    groups: List[CategoryGroup]

remove_duplicate_category = """
You are a support case analyst.
You will be provided a list of cluster items, each with a cluster_id and a category (description of the issue).

Your task:
- Group the relevant User Access and Permissions Management items into the following two categories:

1. **User Account Creation**
   - Includes: new user creation, user setup, user onboarding, user account setup, account creation in Aptean connect, User Accounts setup management, User Account creation requests, User Account setup issues, etc.
2. **User Access and Credentials**
   - Includes: login failures, session lockouts, password resets issues, password reset requests, credential issues, authentication problems, permission changes, deactivation, role assignment, granting and revoking permissions, permissions setup, modifying user roles, inconsistent user roles or permissions, passwords expire, permissions management, modifying user roles etc.
   
Return a valid JSON object **strictly** in this format:
{{
    "User Account Creation": [<Input Category Topics>],
    "User Access and Credentials": [<Input Category Topics>]
}}
 
Do not include explanations — only the valid JSON output.
 
Input:
{cluster_data}
"""

def clean_control_chars(text):
    return re.sub(r'[\x00-\x1F\x7F]', '', text or '')

def categor_format_input_data(data):
    return "\n".join([f"{cid}: {desc}" for cid, desc in data])

def embed_case(case_df):
    case_list = case_df['combined_case_input'].tolist()
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed all case summaries
    case_embeddings = []
    for case in tqdm(case_list, desc="Embedding cases"):
        embedding = embedder.encode([case])[0]
        case_embeddings.append(embedding)

    case_embeddings = np.array(case_embeddings, dtype=np.float32)

    # Normalize embeddings (L2 normalization)
    normalized_embeddings = normalize(case_embeddings)

    return normalized_embeddings

def cluster_case(case_df):
    """
    Clusters embedded case descriptions using KMeans and appends cluster IDs to the DataFrame.
    """
    size = len(case_df)

    if size <= 5000:
        n_clusters = 150
    elif size <= 10000:
        n_clusters = 300
    elif size <= 15000:
        n_clusters = 400
    else:
        n_clusters = 500

    case_embeddings = embed_case(case_df)

    clusterer = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=42,
        n_init='auto'
    )

    cluster_predictions = clusterer.fit_predict(case_embeddings)
    case_df['cluster'] = cluster_predictions

    return case_df

def get_case_info(row):
    case_info_str = ""
    col_name_tag_list = [
        ('subject', 'SUBJECT'),
        ('description', 'DESCRIPTION'),
        ('posts', 'POSTS'),
        ('emails', 'EMAILS')
    ]

    for col_name, col_tag in col_name_tag_list:
        val = row.get(col_name)
        if pd.notna(val):
            case_info_str += f"<{col_tag}>\n{val}\n</{col_tag}>\n\n"

    return case_info_str

def process_cluster(cluster_id, cluster_df):
    # Prepare example case string
    separator_line = "-" * 25
    case_divider = f"\n{separator_line}\n\n"
    num_examples = min(25, len(cluster_df))

    sampled_df = cluster_df.sample(num_examples)
    example_case_list = sampled_df['case_info'].tolist()
    case_numbers = cluster_df['case_number'].tolist()
    example_cluster_cases_str = case_divider.join(example_case_list)

    # Format classification prompt
    case_classification_prompt = case_prompt_template.format(
        categories_info=categories_info,
        tag_info=tag_info,
        case_info=example_cluster_cases_str
    )

    message_history = [{
        "role": "user",
        "content": [{"type": "text", "text": case_classification_prompt}]
    }]

    case_classification_dict = {}
    try:
        case_classification = adaptor.chat.completions.create(
            pydantic_model=CaseClassification,
            num_retries=25,
            model="gpt-4o-mini",
            messages=message_history,
            max_tokens=4096,
            stream=False
        )

        if case_classification:
            case_classification_dict = case_classification.model_dump()
            case_classification_dict['customer_query'] = clean_control_chars(case_classification_dict.get('customer_query',''))
            case_classification_dict['case_summary'] = clean_control_chars(case_classification_dict.get('case_summary',''))
            case_classification_dict['case_root_cause'] = clean_control_chars(case_classification_dict.get('case_root_cause',''))
            case_classification_dict['custcase_recommendationomer_query'] = clean_control_chars(case_classification_dict.get('case_recommendation',''))

            case_classification_dict['case_numbers'] = case_numbers
            case_classification_dict['case_count'] = len(case_numbers)
            case_classification_dict['cluster_cases_str'] = example_cluster_cases_str

    except Exception as e:
        print(f"process_cluster - adaptor call failed: {e}")
        traceback.print_exc()

    return cluster_id, case_classification_dict

def remove_duplicate_cluster(cluster_case_classification_analysis):
    admin_issue_list = []

    # Collect clusters labeled as 'Admin Issue'
    for cluster_id, case_dict in cluster_case_classification_analysis.items():
        if case_dict.get('insight_category') == 'Admin Issue':
            admin_issue_list.append((cluster_id, case_dict.get('lv3', '')))

    print("Total Admin Count - ", len(admin_issue_list))
    
    formatted_data = categor_format_input_data(admin_issue_list)
    remove_duplicate_category_prompt = remove_duplicate_category.format(cluster_data=formatted_data)

    print("remove_duplicate_category_prompt -", len(remove_duplicate_category_prompt))

    message_history = [{
        "role": "user",
        "content": [{"type": "text", "text": remove_duplicate_category_prompt}]
    }]

    cluster_grouping_response = None
    try:
        cluster_grouping_response = adaptor.chat.completions.create(
            pydantic_model=ClusterGroupingResponse,
            num_retries=10,
            model="gpt-4o",
            messages=message_history,
            max_tokens=4096,
            stream=False
        )
    except Exception as e:
        print(f"remove_duplicate_cluster - grouping adaptor call error: {e}")
        traceback.print_exc()

    if not cluster_grouping_response:
        return cluster_case_classification_analysis

    category_groups = cluster_grouping_response.model_dump().get("groups", [])
    print("category_groups-", category_groups)

    for category_group in category_groups:
        cluster_ids = category_group.get("cluster_ids", [])
        lv3_category_name = category_group.get("category_name", '')
        if not cluster_ids:
            continue

        print("Removed cluster Ids size", len(cluster_ids))

        cluster_case_numbers = []
        selected_cluster_data = None
        selected_clusder_id=0
        for cluster_id in cluster_ids:
            case_data = cluster_case_classification_analysis.pop(cluster_id, None)
            if case_data:
                if not selected_cluster_data:
                    selected_cluster_data = case_data
                    selected_clusder_id = cluster_id                    
                cluster_case_numbers.extend(case_data.get("case_numbers", []))

        if selected_cluster_data:
            selected_cluster_data['case_numbers'] = cluster_case_numbers
            selected_cluster_data['case_count'] = len(cluster_case_numbers)
            cluster_case_classification_analysis[selected_clusder_id] = selected_cluster_data

    return cluster_case_classification_analysis

def analysis_process(file_name):
    try:
        file_path = '/home/ec2-user/kathiravan'
        input_file_name = f"{file_path}/input/{file_name}"
        
        # Load data
        df = pd.read_excel(input_file_name)
        #df = df.sample(n=1000, random_state=42)  # Limit rows for analysis
        #df = df[df['Status'] == 'Closed']  # Only closed cases
        
        # Traverse Global
        #df = df[df['Customer Asset'].str.contains('OSAS', case=False, na=False)]
        #df = df[df['Customer Asset'].str.contains('Traverse', case=False, na=False)]

        # ProcessPro
        #df = df[df['Customer Asset'].str.contains('Global', case=False, na=False)]
        #df = df[df['Customer Asset'].str.contains('Premier', case=False, na=False)]

        #Gould Hall
        #df = df[~df['Account Name'].str.contains('Headlam', case=False, na=False)]

        #Apprise
        #df = df[~df['Service Team'].str.contains('Apprise-EDI', case=False, na=False)]

        # Just Food & bc food
        datetime_format = "%m/%d/%Y %I:%M %p"
        df['datetime_opened_dt'] = pd.to_datetime(df['Date/Time Opened'], format=datetime_format, errors='coerce')

        df = df[
            (df['Status'] == 'Closed') &
            (df['datetime_opened_dt'] >= pd.Timestamp("2025-01-01"))
        ]

        df_copy = df.copy()

        print(f"Size - {len(df)}")

        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(' ', '_')
              .str.replace('/', '')
              .str.replace(':', '')
        )

        df['case_info'] = df.apply(get_case_info, axis=1)
        df['combined_case_input'] = df[['subject', 'description', 'resolution_summary']].fillna('').agg('\n'.join, axis=1)
        case_cluster_df = cluster_case(df)
        
        cluster_case_classification_analysis = {}

        case_cluster_df_groupby = case_cluster_df.groupby('cluster')
        print(f"Initial Cluster size -- {len(case_cluster_df_groupby)}")

        # Parallel cluster classification
        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(process_cluster, cluster_id, cluster_df)
                for cluster_id, cluster_df in case_cluster_df_groupby
            ]
            for future in as_completed(futures):
                cluster_id, result = future.result()
                if result:
                    cluster_case_classification_analysis[cluster_id] = result
    
        # Deduplicate Admin Issue clusters
        cluster_case_classification_analysis = remove_duplicate_cluster(cluster_case_classification_analysis)
        cluster_case_classification_analysis = remove_duplicate_cluster(cluster_case_classification_analysis)

        print(f"Final Cluster size -- {len(cluster_case_classification_analysis)}")
        
        # Enrich each cluster with statistics
        for cluster_id, case_dict in cluster_case_classification_analysis.items():
            cluster_df = df[df['case_number'].isin(case_dict['case_numbers'])].copy()

            # Convert dates
            datetime_format = "%m/%d/%Y %I:%M %p"
            cluster_df['datetime_opened'] = pd.to_datetime(cluster_df['datetime_opened'], format=datetime_format, errors='coerce')
            cluster_df['datetime_closed'] = pd.to_datetime(cluster_df['datetime_closed'], format=datetime_format, errors='coerce')

            # Calculate resolution days
            cluster_df['resolution_days'] = (cluster_df['datetime_closed'] - cluster_df['datetime_opened']).dt.total_seconds() / 86400
            if 'resolution_days' in cluster_df and not cluster_df['resolution_days'].isna().all():
                case_dict['avg_resolution_days'] = round(cluster_df['resolution_days'].mean(), 2)
                case_dict['median_resolution_days'] = round(cluster_df['resolution_days'].median(), 2)
                case_dict['p95_resolution_days'] = round(np.percentile(cluster_df['resolution_days'].dropna(), 95), 2)
            else:
                case_dict['avg_resolution_days'] = case_dict['median_resolution_days'] = case_dict['p95_resolution_days'] = ""

            # Satisfaction score
            satisfaction_col = 'overall_satisfaction' if 'overall_satisfaction' in cluster_df.columns else 'overall_atisfaction'
            if satisfaction_col in cluster_df and not cluster_df[satisfaction_col].isna().all():
                case_dict['avg_satisfaction_score'] = round(cluster_df[satisfaction_col].mean(), 2)
                case_dict['median_satisfaction_score'] = round(cluster_df[satisfaction_col].median(), 2)
                case_dict['p25_satisfaction_score'] = round(np.percentile(cluster_df[satisfaction_col].dropna(), 25), 2)
                case_dict['p95_satisfaction_score'] = round(np.percentile(cluster_df[satisfaction_col].dropna(), 95), 2)
            else:
                case_dict['avg_satisfaction_score'] = case_dict['median_satisfaction_score'] = \
                case_dict['p25_satisfaction_score'] = case_dict['p95_satisfaction_score'] = ""

            # Customer distribution
            if 'account_name' in cluster_df.columns:
                cust_counts = cluster_df['account_name'].dropna().value_counts()
                case_dict['ticket_distribution'] = "; ".join(f"{name}:{count}" for name, count in cust_counts.items())
                case_dict['top_5_customers'] = "; ".join(f"{name}:{count}" for name, count in cust_counts.head(5).items())
            else:
                case_dict['ticket_distribution'] = case_dict['top_5_customers'] = ""

            # Severity distribution
            if 'case_severity' in cluster_df.columns:
                sev_counts = cluster_df['case_severity'].dropna().value_counts()
                case_dict['case_severity_distribution'] = "; ".join(f"{sev}:{count}" for sev, count in sev_counts.items())
            else:
                case_dict['case_severity_distribution'] = ""

            # Product versions
            versions = cluster_df.get('product_version_name', pd.Series()).dropna().unique().astype(str).tolist()
            case_dict['product_version_name'] = ', '.join(versions) if versions else ""

        # Generate KB summaries
        for case_dict in cluster_case_classification_analysis.values():
            case_dict.update({
                "kb_status": "",
                "kb_article_number": "",
                "kb_title": "",
                "kb_problem": "",
                "kb_solution": ""
            })

            if case_dict.get('kb_candidate') == 'Y':
                case_numbers = case_dict.get('case_numbers', [])

                filtered_df = df[df['case_number'].isin(case_numbers)]
                filtered_df = filtered_df.drop_duplicates(subset='article_number')
                
                required_cols = ['article_number', 'title', 'problem', 'resolution']
                filtered_df = filtered_df[
                    filtered_df[required_cols].notna().all(axis=1) &
                    (filtered_df[required_cols] != '').all(axis=1)
                ]

                kb_article = None
                try:
                    result_list = filtered_df.to_dict(orient='records') if not filtered_df.empty else None
                    kb_article = kb_analysis(case_dict, result_list) if result_list else kb_analysis(case_dict)
                except Exception as e:
                    print(f"Generate KB summaries - error: {e}")
                    traceback.print_exc()

                if kb_article:
                    raw_article_number = kb_article.get('article_number', None)
                    article_number = str(raw_article_number) if raw_article_number is not None else ""

                    if article_number == "" or len(article_number) <= 1:
                        case_dict["kb_status"] = "New KB"
                        article_number = ""
                    else:
                        case_dict["kb_status"] = "Updating existing KB"

                    case_dict["kb_article_number"] = article_number
                    case_dict["kb_title"] = clean_control_chars(kb_article.get('title', ''))
                    case_dict["kb_problem"] = clean_control_chars(kb_article.get('problem', ''))
                    case_dict["kb_solution"] = clean_control_chars(kb_article.get('solution', ''))
            else:
                case_dict["kb_status"] = "No KB recommendation"

        # Save Excel output
        product_name = df['product_line'].iloc[0]
        output_file_name = f"{file_path}/output/{product_name}_AI Analysis.xlsx"

        result_df = pd.DataFrame.from_dict(cluster_case_classification_analysis, orient='index').reset_index(drop=True)
        result_df.drop(columns=['cluster_cases_str'], errors='ignore', inplace=True)

        with pd.ExcelWriter(output_file_name, engine='openpyxl') as writer:
            # Full detailed output
            custom_column_names = {
                'lv1': 'LV1',
                'lv2': 'LV2',
                'lv3': 'LV3',
                'insight_category': 'Insight Category',
                'case_summary': 'Case Summary',
                'case_root_cause': 'Root Cause',
                'case_recommendation': 'Recommendation',
                'case_numbers': 'Case Numbers',
                'case_count': 'Case Count',
                'product_version_name': 'Product Version Name',
                'avg_resolution_days': 'Average Resolution Days',
                'median_resolution_days': 'Median Resolution Days',
                'p95_resolution_days': '95th Percentile Resolution Days',
                'avg_satisfaction_score': 'Average Satisfaction Score',
                'median_satisfaction_score': 'Median Satisfaction Score',
                'p25_satisfaction_score': '25th Percentile Satisfaction Score',
                'p95_satisfaction_score': '95th Percentile Satisfaction Score',
                'ticket_distribution': 'Customer Ticket Distribution',
                'top_5_customers': 'Top-5 Customers by Tickets',
                'case_severity_distribution': 'Case Severity Distribution',
                "kb_candidate": 'Knowledge Base Candidate',
                'kb_article_number': 'KB Article Number',
                'kb_status': 'KB Status',
                'kb_title': 'KB Title',
                'kb_problem': 'KB Problem',
                'kb_solution': 'KB Solution'
            }

            detailed_output_df = result_df.rename(columns=custom_column_names)
            detailed_output_df.to_excel(writer, index=False, sheet_name="Detailed Output")

            # Per-insight category summaries
            insight_categories = result_df['insight_category'].dropna().unique().tolist()
            insight_categories = [cat for cat in insight_categories if str(cat).strip() and cat != "Other"]

            for category in insight_categories:
                category_df = result_df[result_df['insight_category'] == category].copy()
                category_df = category_df[
                    [
                        'lv3', 'case_summary', 'case_root_cause', 'case_recommendation', 'case_numbers',
                        'case_count', 'product_version_name', 'avg_resolution_days', 'median_resolution_days',
                        'avg_satisfaction_score', 'median_satisfaction_score',
                        'ticket_distribution', 'top_5_customers', 'case_severity_distribution'
                    ]
                ].sort_values(by='case_count', ascending=False)

                category_df.rename(columns={
                        'lv3': category,
                        'case_summary': 'Case Summary',
                        'case_root_cause': 'Root Cause',
                        'case_recommendation': 'Recommendation',
                        'case_numbers': 'Case Numbers',
                        'case_count': 'Case Count',
                        'product_version_name': 'Product Version Name',
                        'avg_resolution_days': 'Average Resolution Days',
                        'median_resolution_days': 'Median Resolution Days',
                        'avg_satisfaction_score': 'Average Satisfaction Score',
                        'median_satisfaction_score': 'Median Satisfaction Score',
                        'ticket_distribution': 'Customer Ticket Distribution',
                        'top_5_customers': 'Top-5 Customers by Tickets',
                        'case_severity_distribution': 'Case Severity Distribution'
                    }, inplace=True)
                category_df.to_excel(writer, index=False, sheet_name=category)

            # KB Candidate sheet
            kb_df = result_df[result_df['kb_candidate'] == 'Y'].copy()
            if not kb_df.empty:
                kb_df = kb_df[
                    [
                        'kb_status', 'kb_article_number', 'kb_title', 'kb_problem', 'kb_solution', 'case_summary', 'case_root_cause',
                        'case_recommendation', 'case_numbers', 'case_count', 'product_version_name',
                        'avg_resolution_days', 'median_resolution_days',
                        'avg_satisfaction_score', 'median_satisfaction_score',
                        'ticket_distribution', 'top_5_customers', 'case_severity_distribution'
                    ]
                ].sort_values(by='case_count', ascending=False)

                kb_df.rename(columns={
                        'kb_status': 'KB Status',
                        'kb_article_number': 'KB Article Number',
                        'kb_title': 'KB Title',
                        'kb_problem': 'KB Problem',
                        'kb_solution': 'KB Solution',
                        'case_summary': 'Case Summary',
                        'case_root_cause': 'Root Cause',
                        'case_recommendation': 'Recommendation',
                        'case_numbers': 'Case Numbers',
                        'case_count': 'Case Count',
                        'product_version_name': 'Product Version Name',
                        'avg_resolution_days': 'Average Resolution Days',
                        'median_resolution_days': 'Median Resolution Days',
                        'avg_satisfaction_score': 'Average Satisfaction Score',
                        'median_satisfaction_score': 'Median Satisfaction Score',
                        'ticket_distribution': 'Customer Ticket Distribution',
                        'top_5_customers': 'Top-5 Customers by Tickets',
                        'case_severity_distribution': 'Case Severity Distribution'
                    }, inplace=True)

                kb_df.to_excel(writer, index=False, sheet_name="Knowledge Base Candidate")

        # Final report trigger
        report_process(product_name, df_copy, output_file_name)
        print("-- Completed --")

    except Exception as e:
        print(f"analysis_process - error: {e}")
        traceback.print_exc()

if __name__ == "__main__":

    #erp_names2 = ["Oxaion ERP","Syncos MES","Swords","Made2Manage","Produce Pro ERP","Ross"]

    # erp_names3 = [
    #     #'Traverse Global',
    #     #'ProcessPro',
    #     #'Paragon_Flexipod',
    #     #'Paragon_HDX',
    #     #'Paragon_Application',
    # ]

    ''' erp_names = [
        "Ramsauer & Sturmer",
        "Affinitus FreshWare",
        "Affinitus GrowMaster",
        "Respond",
        "Prima Solutions ERP",
        "Affinitus ChefServe",
        "Elucid",
        "3T Logistics",
        "Patch OEE",
        "Lascom PLM",
        "API Pro",
        "TOTALogistix",
        "RLM ERP",
        "Full Circle ERP",
        "WorkWise ERP",
        "JustFood",
        "Momentis Systems",
        "Catalyst",
        "Aptean Food and Beverage",
        "Exenta ERP",
        "AssetPoint",
        "EDI Direct",
        "Aptean Business Solutions",
        "Apparel Business Systems",
        "EquipSoft",
        "Foodware BC",
        "Factory",
        "Aptean EAM",
        "LINKFRESH 365 Business Central",
        "Drink-IT",
        "bc Food",
        "UnityF8",
        "Cimdata ERP",
        "Proteus",
        "Calidus",
        "Logis ERP",
        "Southware",
        "Intuitive",
        "Aptean Retail Planning",
        "Master Distribution System",
        "Aptean Retail PLM",
        "Global Service",
        "Impress",
        "irms360",
        "Encompix",
        "ImPuls",
        "trend SWM"
    ]'''
    #erp_names = ["Impress","WorkWise ERP","Unity","Apprise"]
    #erp_names =["Apprise"]
    
    erp_names = [
        #"Impress",
        #"WorkWise ERP",
        'bc Food',
        'JustFood'
    ]
    
    for erp_name in erp_names:
        try:
            file_name = f"{erp_name}_case_list.xlsx"
            print(f"\n--- Starting analysis for: {file_name} ---")
            analysis_process(file_name)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # product_name = 'Made2Manage'
    # file_name = f"{product_name}_case_list.xlsx"
    # file_path = '/home/ec2-user/kathiravan'
    # input_file_name = f"{file_path}/input/{file_name}"
    
    # # Load data
    # df = pd.read_excel(input_file_name)
    # #df = df.head(1000)  # Limit rows for analysis
    # df = df[df['Status'] == 'Closed']  # Only closed cases
    # #df = df[df['Customer Asset'].str.contains('ProcessPro Premier', case=False, na=False)]

    # df_copy = df.copy()
    # report_process(product_name, df_copy, "/home/ec2-user/kathiravan/output/Made2Manage_AI Analysis.xlsx")
