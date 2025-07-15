import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import json

from concurrent.futures import ProcessPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer
from typing import  Literal
from pydantic import BaseModel, Field
from sklearn.cluster import HDBSCAN 

from PydanticAdaptorOpenRouter import PydanticAdaptorOpenRouter
from knowledgebase_analysis import kb_analysis 
from summary_report_generate import reprot_process

tag_info = """Here is some context on the different pieces of information you may be given
access to as part of analysing the case. Each of the below menitoned tags may or may not be
present for each case

<TAG INFO>
1. DESCRIPTION - This represents the description of the case and some notes around it, written by the internal support team while discussing the case amongst themselves as they work to resolve it. These notes are typically internal-facing and may contain technical jargon or references to known issues or configurations.
2. POSTS - This represents ad-hoc, often unstructured information present in internal logs, chat notes, or tracking documents as the support team investigates the issue. These entries may contain useful insights, but also a fair amount of noise, such as meeting schedules, team coordination details, or status updates that are not directly relevant to the technical root cause.
3. EMAILS - This section contains the chronological conversation between the **support agent(s)** and the **customer**. It reflects the full communication trail starting from the customer's first report of the issue through to resolution or closure.
   These communications are often informal, and may include misspellings, mixed technical and non-technical language, and emotional tone (e.g., customer frustration or urgency). They are invaluable for understanding the **true impact of the issue**, the **steps taken**, and **what actually resolved it**.
</TAG INFO></TAG INFO>
"""

case_prompt_template = """You are a support analyst at Aptean reviewing a subset of technical support cases that have been grouped together 
due to their similar issue patterns and underlying themes. Your task is to analyze the commonalities across these cases and propose a clear, 
descriptive topic that accurately summarizes the core problem or theme represented by this cluster. Provide a detailed explanation of why this 
topic best represents the pattern observed in these cases.

Here is some information regarding the categories available to you
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

- If the category is `Knowledge Base Candidate`:  
  *Describe in detail what the knowledgebase article should cover.*  
  Include:
  - A clear and relevant title
  - A concise problem description
  - A step-by-step resolution or workaround
  - Optional: preconditions, UI paths, screenshots, or configuration notes  
  The article should help customers self-resolve the issue and reduce ticket volume.

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

    - Admin Issue: If an issue is involves admin related stuff such as adding users, removing users, ip whitelisting
    or other admin related tasks, then classify them as an admin issue.

     - Data Error: If an issue is directly related to data corruption issue or any form of data issue but is not related to any integration issue, 
    feature enhancement, any module specific issue or any other technical issue then classify it as a Data Error.

    - Setup Issue: If an issue is directly related to a system setup issue, or a product configuration issue but is not related to any integration issue, data corruption issue, feature enhancement, 
    any module specific issue or any other technical issue then classify it as a Setup Issue.

    - Process Gaps: If an issue is directly not related to any issue but is more like a process gap such as Delays in responding or resolving issues, 
    Incorrect team or priority assignment, Premature case closure or repeat reopenings, Customers needing to follow up multiple times, 
    Support policies or SLAs not followed properly etc, then classify it as a Process Gap

    - Educational Issue: If an issue is directly not related to any issues, process gaps in support process or cannot be addressed by a simple knowledge base article, 
    and a tutorial that is properly educational could be better suited for the given issue or situation, then classify it as an educational issue.

    - Knowledge Base Candidate: If an issue is related to Repeated questions by user, How-to queries, Troubleshooting steps that were recommended to them, 
    any gaps found in existing Knowledge Base Articles or documentation, or Resolution steps that could be reused then classify as a KB Candidate. 
    It should not be related to any integration issue, data corruption issue, feature enhancement, any module specific issue, setup issue or 
    any other technical issue and should really be related to knowledge base article.

    - Other: If an issue does not follow in any of these categories. Classify it as other. When you classify an issue
    as other. Provide a name for a category that should be created that will represent the class of issues represented
    by this current issue. Only do this if the issue cannot be clearly placed in any of the above mentioned categories.
    Also, provide reasoning behinf why it could not be placed in any of the above mentioned categories and why the suggested
    name is a good category name for the class of issues represented by the current issue
"""

adaptor = PydanticAdaptorOpenRouter(openai_client=None, openai_api_key=os.getenv('OPENROUTER_API_KEY'))

class CaseClassification(BaseModel):
    LV1: str = Field(..., description="Broad functional area such as Access, Reporting, Integration, Admin, Very high level modules etc. There should not be duplicates within this group and should not be duplicates across  LV2 and LV3 groups.")

    LV2: str = Field(..., description="Specific product modules, workflow, or component within the product. There should not be duplicates within this group and should not be duplicates across  LV1 and LV3 groups.")
 
    LV3: str = Field(..., description="Concrete technical symptoms or failure pattern (e.g., 'Login screen hangs', 'Incorrect totals in report', 'Specific Login Failures', 'Specific Product modules', 'Specific Data Discrepancies', 'Specific Setup issues', 'Specific Errors',  ). There should not be duplicates within this group and should not be duplicates across  LV1 and LV2 groups.")
    
    insight_category: Literal['Integration', 'Software Bug', 'Feature Enhancement', 'Module Issue', 'Admin Issue', 'Data Error', 'Setup Issue', 'Process Gap', 'Educational Issue', 'Knowledge Base Candidate', 'Other'] = Field(
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
        description="Clearly explain the root cause of the issue across the analyzed cases. Your explanation should be aligned with one of the following categories: Integration, Software Bug, Feature Enhancement, Module Issue, Admin Issue, Data Error, Setup Issue, Process Gap, Educational Issue, Knowledge Base Candidate, or Other. Be specific about what failed technically or procedurally."
    )
    case_recommendation: str = Field(
        ..., 
        description="Based on the root cause and the insight category, provide very detailed actionable recommendations as to what exactly should be done to address all such issues. These may include one of the following- fixing a software bug, proposing a feature change or enhancement, fixing a specific data issue, addressing an integration issue,  preventing a potential configuration or setup issue , closing a process gap, automating a task, adding Knowledgebase content, or initiating training. Your recommendation should succinct and should clearly correspond to the category."
    )

# def embed_case(case_df):
#     case_list = case_df['case_info'].tolist()

#     #embed all case summaries
#     embedder = SentenceTransformer("all-MiniLM-L6-v2")
#     case_embeddings = []
#     for case in tqdm(case_list):
#         case_embedding = embedder.encode([case])[0]
#         case_embeddings.append(case_embedding)

#     case_embeddings = np.array(case_embeddings)

#     return case_embeddings

# def cluster_case(case_df):
#     case_embeddings = embed_case(case_df)
#     clusterer = HDBSCAN(min_cluster_size=2)
#     cluster_predictions = clusterer.fit_predict(case_embeddings)
#     case_df['cluster'] = cluster_predictions
#     return case_df

def embed_case(case_df):
    case_list = case_df['case_info'].tolist()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    case_embeddings = [embedder.encode([case])[0] for case in tqdm(case_list)]
    return np.array(case_embeddings)

def cluster_case(case_df, initial_min_cluster_size=2, max_cluster_size=100):
    case_embeddings = embed_case(case_df)

    # Initial clustering
    clusterer = HDBSCAN(min_cluster_size=initial_min_cluster_size)
    cluster_predictions = clusterer.fit_predict(case_embeddings)
    case_df['cluster'] = cluster_predictions

    new_clusters = []
    cluster_offset = 0  # for re-assigning cluster numbers without overlap

    for cluster_id in sorted(case_df['cluster'].unique()):
        if cluster_id == -1:
            # Noise points: keep as is
            noise_df = case_df[case_df['cluster'] == -1].copy()
            noise_df['cluster'] = -1
            new_clusters.append(noise_df)
            continue

        cluster_group = case_df[case_df['cluster'] == cluster_id].copy()
        if len(cluster_group) <= max_cluster_size:
            # Cluster is small enough, keep as is
            cluster_group['cluster'] = cluster_id + cluster_offset
            new_clusters.append(cluster_group)
        else:
            # Re-cluster this group
            print(f"Re-clustering cluster {cluster_id} with {len(cluster_group)} points...")
            sub_embeddings = embed_case(cluster_group)
            sub_clusterer = HDBSCAN(min_cluster_size=initial_min_cluster_size)
            sub_preds = sub_clusterer.fit_predict(sub_embeddings)

            # Offset sub-cluster labels
            sub_cluster_ids = set(sub_preds)
            for sub_id in sub_cluster_ids:
                sub_cluster = cluster_group[sub_preds == sub_id].copy()
                if sub_id == -1:
                    sub_cluster['cluster'] = -1
                else:
                    sub_cluster['cluster'] = cluster_offset
                    cluster_offset += 1
                new_clusters.append(sub_cluster)

    result_df = pd.concat(new_clusters, ignore_index=True)
    return result_df

def get_case_info(row):
    case_info_str = ""

    col_name_tag_list = [('subject', 'DESCRIPTION'), ('posts', 'POSTS'), ('emails', 'EMAILS')]
    for (col_name, col_tag) in col_name_tag_list:
        val = row.loc[col_name]
        if not pd.isna(val):
            case_info_str += f"<{col_tag}>\n{val}\n</{col_tag}>\n\n"

    return case_info_str

def process_cluster(cluster_id, cluster_df):
    print(f"Processing started - {cluster_id}")

    separator_line = "-" * 25
    case_divider = f"\n{separator_line}\n\n"
    num_example_cases_per_cluster = 25

    sampled_cluster_df = cluster_df.sample(min(num_example_cases_per_cluster, len(cluster_df)))
    example_case_list = sampled_cluster_df['case_info'].tolist()
    case_numbers = cluster_df['case_number'].tolist()
    example_cluster_cases_str = case_divider.join(example_case_list)

    case_classification_prompt = case_prompt_template.format(
        categories_info=categories_info,
        tag_info=tag_info,
        case_info=example_cluster_cases_str
    )

    content = [{"type": "text", "text": case_classification_prompt}]
    curr_message = {
        "role": "user",
        "content": content
    }

    message_history = [curr_message]
    case_classification = None
    case_classification_dict=dict()
    
    try:
        case_classification = adaptor.chat.completions.create(
            pydantic_model=CaseClassification,
            num_retries=10,
            model="gpt-4o-mini",
            messages=message_history,
            max_tokens=4096,
            stream=False
        )
    except Exception as e:
        print(e)

    if case_classification:
        case_classification_dict = case_classification.model_dump()

    case_classification_dict['case_numbers'] = case_numbers
    case_classification_dict['case_count'] = len(case_numbers)

    # Convert datetime columns
    datetime_format = "%m/%d/%Y %I:%M %p"
    cluster_df['datetime_opened'] = pd.to_datetime(cluster_df['datetime_opened'], format=datetime_format, errors='coerce')
    cluster_df['datetime_closed'] = pd.to_datetime(cluster_df['datetime_closed'], format=datetime_format, errors='coerce')

    # Compute resolution days
    cluster_df['resolution_days'] = (cluster_df['datetime_closed'] - cluster_df['datetime_opened']).dt.total_seconds() / 86400

    # Resolution days stats
    if 'resolution_days' in cluster_df.columns:
        if not cluster_df['resolution_days'].isna().all():
            resolution_days_mean   = round(cluster_df['resolution_days'].mean(), 2)
            resolution_days_median = round(cluster_df['resolution_days'].median(), 2)
            resolution_days_95p    = round(np.percentile(cluster_df['resolution_days'].dropna(), 95), 2)
        else:
            resolution_days_mean = resolution_days_median = resolution_days_95p = "N/A"
    else:
        resolution_days_mean = resolution_days_median = resolution_days_95p = "N/A"

    # Satisfaction score stats (corrected column name spelling if needed)
    satisfaction_col = 'overall_satisfaction' if 'overall_satisfaction' in cluster_df.columns else 'overall_atisfaction'
    if satisfaction_col in cluster_df.columns:
        if not cluster_df[satisfaction_col].isna().all():
            sat_score_mean   = round(cluster_df[satisfaction_col].mean(), 2)
            sat_score_median = round(cluster_df[satisfaction_col].median(), 2)
            sat_score_25p    = round(np.percentile(cluster_df[satisfaction_col].dropna(), 25), 2)
            sat_score_95p    = round(np.percentile(cluster_df[satisfaction_col].dropna(), 95), 2)
        else:
            sat_score_mean = sat_score_median = sat_score_25p = sat_score_95p = "N/A"
    else:
        sat_score_mean = sat_score_median = sat_score_25p = sat_score_95p = "N/A"

    # Customer distribution
    if "account_name" in cluster_df.columns:
        cust_counts = cluster_df["account_name"].dropna().value_counts()
        customer_dist_str = "; ".join(f"{name}:{count}" for name, count in cust_counts.items())
        top5 = cust_counts.head(5)
        top5_str = "; ".join(f"{name}:{count}" for name, count in top5.items())
    else:
        customer_dist_str = "N/A"
        top5_str = "N/A"

    # Case severity distribution
    if "case_severity" in cluster_df.columns:
        sev_counts = cluster_df["case_severity"].dropna().value_counts()
        severity_dist_str = "; ".join([f"{sev}:{n}" for sev, n in sev_counts.items()])
    else:
        severity_dist_str = "N/A"

    # Production Version
    versions = cluster_df['product_version_name'].dropna().unique().tolist() \
               if 'product_version_name' in cluster_df.columns else []

    # Final dictionary update
    case_classification_dict['product_version_name'] = ', '.join(versions) or "N/A"
    case_classification_dict['avg_resolution_days'] = resolution_days_mean
    case_classification_dict['median_resolution_days'] = resolution_days_median
    case_classification_dict['p95_resolution_days'] = resolution_days_95p
    case_classification_dict['avg_satisfaction_score'] = sat_score_mean
    case_classification_dict['median_satisfaction_score'] = sat_score_median
    case_classification_dict['p25_satisfaction_score'] = sat_score_25p
    case_classification_dict['p95_satisfaction_score'] = sat_score_95p
    case_classification_dict['ticket_distribution'] = customer_dist_str or "N/A"
    case_classification_dict['top_5_customers'] = top5_str or "N/A"
    case_classification_dict['case_severity_distribution'] = severity_dist_str or "N/A"
    return cluster_id, case_classification_dict

def analysis_process(file_name):
    try:
        file_path = '/home/ec2-user/kathiravan'
        input_file_name = f"{file_path}/input/{file_name}"
        
        df = pd.read_excel(input_file_name)
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('/', '')
            .str.replace(':', '')
        )

        # Filter first 100 count
        #df = df[0:500]

        # Closed case only
        df = df[df['status'] == 'Closed']

        df['case_info'] = df.apply(get_case_info, axis=1)
        case_cluster_df = cluster_case(df)

        case_cluster_df_groupby = case_cluster_df.groupby('cluster')

        print(f"Clouster size -- {len(case_cluster_df_groupby)}")

        # Parallel execution
        cluster_case_classification_analysis = {}

        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(process_cluster, cluster_id, cluster_df)
                for cluster_id, cluster_df in case_cluster_df_groupby
            ]

            for future in as_completed(futures):
                cluster_id, result = future.result()
                cluster_case_classification_analysis[cluster_id] = result

        for cluster_id, case_dict in cluster_case_classification_analysis.items():
            case_dict["kb_article_number"] = ""
            case_dict["kb_title"] = ""
            case_dict["kb_problem"] = ""
            case_dict["kb_solution"] = ""

            if 'insight_category' in case_dict and case_dict['insight_category'] == 'Knowledge Base Candidate':
                case_numbers = case_dict.get('case_numbers', [])

                # Filter rows with matching case numbers
                filtered_df = df[df['case_number'].isin(case_numbers)]

                # Filter rows where KB fields are NOT None or empty
                required_cols = ['article_number', 'title', 'problem', 'resolution']
                filtered_df = filtered_df[
                    filtered_df[required_cols].notna().all(axis=1) &
                    (filtered_df[required_cols] != '').all(axis=1)
                ]

                kb_article = None
                if not filtered_df.empty:
                    result_list = filtered_df.to_dict(orient='records')
                    print(f"//{result_list}")
                    kb_article = kb_analysis(case_dict, result_list)
                else:
                    kb_article = kb_analysis(case_dict)

                # Extract output fields from result (single KB article model)
                if kb_article:
                    case_dict["kb_article_number"] = kb_article.get('article_number', '')
                    case_dict["kb_title"] = kb_article.get('title', '')
                    case_dict["kb_problem"] = kb_article.get('problem', '')
                    case_dict["kb_solution"] = kb_article.get('solution', '')

        # Product Line Name
        product_name=df['product_line'].tolist()[0]

        # Output file name
        output_file_name = f"{file_path}/output/{product_name}_AI Analysis.xlsx"

        result_df = pd.DataFrame.from_dict(cluster_case_classification_analysis, orient='index').reset_index()
        result_df = result_df.drop(columns=["index"])

        # Save both sheets in one Excel file
        with pd.ExcelWriter(output_file_name, engine='openpyxl') as writer:
            # Sheet 1: Full cluster details
            custom_column_names = {
                'LV1': 'LV1',
                'LV2': 'LV2',
                'LV3': 'LV3',
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
                'kb_article_number': 'KB Article Number',
                'kb_title': 'KB Title',
                'kb_problem': 'KB Problem',
                'kb_solution': 'KB Solution'
            }

            # Apply column renaming
            detailed_output_df = result_df.rename(columns=custom_column_names)
            detailed_output_df.to_excel(writer, index=False, sheet_name="Detailed Output")
            
            insight_categories = result_df['insight_category'].unique().tolist()
            for insight_category in  insight_categories:
                if insight_category == "Knowledge Base Candidate":
                    
                    # KB Candidate Summary
                    kb_df = result_df[result_df['insight_category'] == 'Knowledge Base Candidate'].copy()
                    kb_filtered_df = kb_df[
                            [
                                'kb_article_number', 'kb_title', 'kb_problem', 'kb_solution', 'case_summary', 'case_root_cause', 'case_recommendation', 'case_numbers',
                                'case_count', 'product_version_name', 'avg_resolution_days', 'median_resolution_days', 'avg_satisfaction_score', 'median_satisfaction_score',
                                'ticket_distribution', 'top_5_customers', 'case_severity_distribution'
                            ]
                        ].sort_values(by='case_count', ascending=False)

                    kb_filtered_df.rename(columns={
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

                    kb_filtered_df.to_excel(writer, index=False, sheet_name="Knowledge Base Candidate")

                elif insight_category != "Other":

                    category_df = result_df[result_df['insight_category'] == insight_category].copy()
                    category_df = category_df[
                            [
                                'LV3', 'case_summary', 'case_root_cause', 'case_recommendation', 'case_numbers',
                                'case_count', 'product_version_name', 'avg_resolution_days', 'median_resolution_days', 'avg_satisfaction_score', 'median_satisfaction_score',
                                'ticket_distribution', 'top_5_customers', 'case_severity_distribution'
                            ]
                        ].sort_values(by='case_count', ascending=False)
                        
                    category_df.rename(columns={
                            'LV3': insight_category,
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

                    category_df.to_excel(writer, index=False, sheet_name=insight_category)

        reprot_process(product_name, input_file_name, output_file_name)
 
        print("-- Completed --")
    except Exception as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":

    #file_name = "Produce Pro ERP_case_list.xlsx"
    #analysis_process(file_name)

    # reprot_process('Produce Pro ERP', '/home/ec2-user/kathiravan/input/Produce Pro ERP_case_list.xlsx',
    #     '/home/ec2-user/kathiravan/output/Produce Pro ERP_AI Analysis.xlsx')

    erp_names = [
        'Apprise',
        'Swords',
        'Intuitive',
        'Logis ERP',
        'Impress',
        'Made2Manage',
        'Ross',
        'Produce Pro ERP',
        'RLM ERP',
        'Full Circle ERP',
        'WorkWise ERP',
        'JustFood',
    ]

    for erp_name in erp_names:
        try:
            file_name = f"{erp_name}_case_list.xlsx"
            analysis_process(file_name)
        except Exception as e:
            print(e)
    