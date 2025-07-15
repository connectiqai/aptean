import pandas as pd
import re
import logging
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed

# === File paths ===
# Load data with dtype enforcement for Case Number and Article Number columns to preserve leading zeros
file_path="/home/ec2-user/MasterData/input/"

case_file = f"{file_path}Case Report_2025-07-03.xlsx"
case_email_file = f"{file_path}Case Email Report_2025-07-03.xlsx"
case_post_file = f"{file_path}Case Post Report_2025-07-03.xlsx"
case_cr_file =  f"{file_path}CR Report_2025-07-03.xlsx"
case_survey_file = f"{file_path}Cases with Survey Feedback_2025-07-03.xlsx"
case_kb_article_file = f"{file_path}Cases with Articles Report_2025-07-03.xlsx"
kb_article = f"{file_path}KB Article Report_2025-07-03.xlsx"

df_case = pd.read_excel(case_file, dtype={"Case Number": str})
df_email = pd.read_excel(case_email_file, dtype={"Case Number": str})
df_post = pd.read_excel(case_post_file, dtype={"Case Number": str})
df_change = pd.read_excel(case_cr_file,  dtype={"Linked Case on Insert": str})
df_survey = pd.read_excel(case_survey_file,  dtype={"Case Number": str})
df_case_kb = pd.read_excel(case_kb_article_file,  dtype={"Case Number": str, "Knowledge Article ID": str})
df_kb = pd.read_excel(kb_article,  dtype={"Knowledge Article ID": str, "Article Number": str})

# === Helper Functions ===
def clean_support_email(text):
    try:
        if not isinstance(text, str):
            return ""
        
        skip_keywords = {
            "external", "https://", "ref:_", "*** case updated with comment ***",
            "click here to view", "[https://", "sent:", "to:", "case details",
            "case #:", "severity:", "latest comment from support",
            "the following case has a new comment from aptean support",
            "to view your case, click here", "thread::", "cc:", "caution:", "tel:",
            "main:", "cell:", "this message is confidential", "office:", "mobile:",
            "phone :", "direct:", "fax:", "www.aptean.com<http://www.aptean.com/>",
            "m:", "follow us on:", "----- reply above this line to send a response -----",
            "this email message, including attachments:",
            "This email address is not monitored for incoming requests.",
            "Please do not attempt to respond to this message.",
            "Initial File Upload",
            "--REPLY above this line to respond--"
        }

        cutoff_markers = {
            "[logo description automatically generated]", "from:",
            "this message was created automatically by the mail system",
            "confidentiality notice:",
            "this message contains confidential information and is intended only for the individual named.",
            "regards,", "thanks,", "thank you,", "best regards,", "warm regards,", "best,",
            "kind regards,", "with regards,", "sincerely,", "cheers,", "best wishes,",
            "yours truly,", "yours faithfully,", "respectfully,", "sent from my iphone",
            "thanks & regards,",
            "please join the zoom meeting using the below link."
        }

        cutoff_patterns = [
            re.compile(r'-{2,}.*original message.*-{2,}', re.IGNORECASE),
            re.compile(r'^[- ]*original message[- ]*$', re.IGNORECASE)
        ]

        patterns = {
            "greeting": re.compile(r'^(hi|hello|dear|hey|greetings|good (morning|afternoon|evening|day))[\s,]+.*$', re.IGNORECASE),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
            'cid_image': re.compile(r'\[cid:[^\]]*\]'),
            'phone': re.compile(r'\b(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})\b'),
            'phone2': re.compile(r'\+?\d{1,2}[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}'),
            'underscore_line': re.compile(r'^_+$'),
            'dash_line': re.compile(r'^-+$'),
            "equal_line":  re.compile(r'={4,}'),
        }

        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()

            if any(marker in lowered for marker in cutoff_markers) or any(p.search(stripped) for p in cutoff_patterns):
                break

            if not stripped or lowered in {",", "|", "a:"}:
                continue

            if any(p.search(stripped) for p in patterns.values()):
                continue

            if any(keyword.lower() in lowered for keyword in skip_keywords):
                continue

            if lowered.startswith(("subject:", "case subject:")):
                continue

            cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines)

    except Exception as e:
        logging.warning(f"clean_support_email error: {e}")
        return text

def clean_description(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return ""

        # Define cutoff markers (lowercase for consistent matching)
        cutoff_markers = {
            "regards,", "thanks,", "thank you,", "best regards,", "warm regards,", "best,",
            "kind regards,", "with regards,", "sincerely,", "cheers,", "best wishes,", "thanks & regards,",
            "yours truly,", "yours faithfully,", "respectfully,", "sent from my iphone",
            "this message contains information that is confidential and/or may be privileged",
            "[Logo, company name  Description automatically generated]",
            "the information in this e-mail is confidential and may be legally privileged."
        }

        # Compile patterns for detecting forwarded or original messages
        cutoff_patterns = [
            re.compile(r'-{2,}.*original message.*-{2,}', re.IGNORECASE),
            re.compile(r'^[- ]*original message[- ]*$', re.IGNORECASE),
        ]

        patterns = {
            "greeting": re.compile(r'^(hi|hello|dear|hey|greetings|good (morning|afternoon|evening|day))[\s,]+.*$', re.IGNORECASE),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
            'cid_image': re.compile(r'\[cid:[^\]]*\]'),
            'phone': re.compile(r'\b(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})\b'),
            'phone2': re.compile(r'\+?\d{1,2}[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}'),
            'underscore_line': re.compile(r'^_+$'),
            'dash_line': re.compile(r'^-+$'),
            "equal_line":  re.compile(r'={4,}'),
        }

        # Keywords to skip entire lines
        skip_keywords = {
            "external", "ref:_", "--REPLY above this line to respond--", "sent:", "subject:"
        }

        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()

            # Skip lines that match skip keywords or cutoff patterns
            if any(keyword in lowered for keyword in skip_keywords):
                continue
            if any(p.search(stripped) for p in cutoff_patterns):
                continue
            if any(marker in lowered for marker in cutoff_markers):
                break
            if any(p.search(stripped) for p in patterns.values()):
                continue

            cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines)

    except Exception as e:
        logging.warning(f"clean_description error: {e}")
        return text
    
def clean_subject(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Normalize the text to avoid case sensitivity issues
    cleaned_text = text.strip()

    # Common replacements with consistent order
    replacements = [
        "[EXTERNAL]",
        "RE:",
        "FW:",
        "Re:",
        "Fwd:",
        "Re: [EXTERNAL] FW:",
        "[EXTERNAL] RE:",
        "[EXTERNAL] FW:",
        "[EXTERNAL] Recall: NOTICE:"
    ]
    
    for item in replacements:
        cleaned_text = cleaned_text.replace(item, "")
    
    return cleaned_text.strip()

# === Main Processing ===
def process(product_name, df_case, df_email, df_post, df_change, df_survey, df_case_kb, df_kb, no_of_cases=0):

    # Filter Made2Manage cases
    #df_case = df_case[(df_case['Product Line'].str.strip().str.lower() == product_name.lower()) &(~df_case['Customer Asset'].str.contains("osas", case=False, na=False))]
    df_case = df_case[df_case['Product Line'].str.strip().str.lower() == product_name.lower()]
    # Add SaaS vs onPrem asset type based on 'Has SaaS Asset' column
    df_case['Asset'] = np.where(df_case['Has SaaS Asset flag'] == True, 'SAAS', 'onPrem')


    if no_of_cases > 0:
        df_case = df_case.sample(n=min(no_of_cases, len(df_case)), random_state=42)

    # Clean column names
    for df in [df_case, df_email, df_post, df_change]:
        df.columns = df.columns.str.strip()

    # Clean text
    df_case['Description'] = df_case['Description'].apply(clean_description)
    df_case['Subject'] = df_case['Subject'].apply(clean_subject)
    
    df_email['Message Date'] = pd.to_datetime(
        df_email['Message Date'], 
        format='%m/%d/%Y %I:%M %p',
        errors='coerce'
    )
    df_post['Created Date'] = pd.to_datetime(
        df_post['Created Date'], 
        format='%m/%d/%Y',
        errors='coerce'
    )
    df_email['Text Body'] = df_email['Text Body'].apply(clean_support_email)
    df_post['Body'] = df_post['Body'].apply(clean_support_email)

    # Sort and group Emails
    df_email_sorted = df_email.sort_values(['Case Number', 'Message Date'])
    df_email_grouped = (
        df_email_sorted
        .groupby('Case Number', as_index=False)
        .agg({
            'Text Body': lambda x: '\n\n'.join(x.dropna().astype(str)),
        })
        .rename(columns={'Text Body': 'Emails'})
    )

    # Sort and group Posts
    df_post_sorted = df_post.sort_values(['Case Number', 'Created Date'])
    df_post_grouped = (
        df_post_sorted
        .groupby('Case Number', as_index=False)
        .agg({
            'Body': lambda x: '\n\n'.join(x.dropna().astype(str))
        })
        .rename(columns={'Body': 'Posts'})
    )

    # Filter and group CR (Closed only)
    df_change.rename(columns={"Linked Case on Insert": "Case Number"}, inplace=True)
    df_change['Case Number'] = df_change['Case Number'].astype(str).str.strip()
    df_change = df_change[df_change['Status'].str.strip().str.lower() == 'closed']

    df_change_grouped = (
        df_change.groupby('Case Number', as_index=False)
        .agg({
            'Development Request: Request #': lambda x: ', '.join(x.dropna().astype(str)),
            'Title': lambda x: '\n\n'.join(x.dropna().astype(str)),
            'Description': lambda x: '\n\n'.join(x.dropna().astype(str)),
            'Status': lambda x: ', '.join(x.dropna().astype(str))
        })
        .rename(columns={
            'Development Request: Request #': 'All Requests',
            'Title': 'Request Titles',
            'Description': 'Request Descriptions',
            'Status': 'Request Statuses',
        })
    )

    # Ensure Case Number column type consistency for merging
    for df in [df_case, df_email_grouped, df_post_grouped, df_change_grouped]:
        df['Case Number'] = df['Case Number'].astype(str).str.strip()

    # Merge core dataframes
    df_merged = df_case \
        .merge(df_email_grouped, on='Case Number', how='left') \
        .merge(df_post_grouped, on='Case Number', how='left') \
        .merge(df_change_grouped, on='Case Number', how='left')

    # Merge KB data
    df_case_kb['Case Number'] = df_case_kb['Case Number'].astype(str).str.strip()
    df_case_kb['Knowledge Article ID'] = df_case_kb['Knowledge Article ID'].astype(str).str.strip()
    df_kb.rename(columns={"Knowledge ID": "Knowledge Article ID"}, inplace=True)
    df_kb['Knowledge Article ID'] = df_kb['Knowledge Article ID'].astype(str).str.strip()

    df_case_kb_merged = pd.merge(
        df_case[['Case Number']].drop_duplicates(),
        df_case_kb[['Case Number', 'Knowledge Article ID']],
        on='Case Number',
        how='left'
    )

    df_case_kb_merged = pd.merge(
        df_case_kb_merged,
        df_kb[['Knowledge Article ID', 'Article Number', 'Title', 'Problem', 'Resolution']],
        on='Knowledge Article ID',
        how='left'
    ).drop_duplicates(subset=['Case Number'])

    df_merged = pd.merge(df_merged, df_case_kb_merged, on='Case Number', how='left')

    # Merge Survey data
    df_survey['Case Number'] = df_survey['Case Number'].astype(str).str.strip()
    survey_columns = [
        'Case Number', 'Question: Problem Diagnosis', 'Question: Product Knowledge',
        'Question: Professionalism', 'Question: Communication',
        'Overall Satisfaction', 'Customer Comments'
    ]
    df_survey_filtered = df_survey[survey_columns].drop_duplicates(subset=['Case Number'])
    df_merged = pd.merge(df_merged, df_survey_filtered, on='Case Number', how='left')

    # Output file
    output_path = f"/home/ec2-user/MasterData/output/{product_name}_case_list.xlsx"
    df_merged.to_excel(output_path, index=False)
    print("✅ Final merged case file saved at:", output_path)

if __name__ == "__main__":
    # erp_systems = [
    #     "Mistral ERP",
    #     "Merlin ERP",
    #     "Gould Hall",
    #     "L.i.S.A Software ERP",
    #     "Principal Logistics Shared",
    #     "WorkWise ERP",
    #     "Momentis Systems",
    #     "Catalyst",
    #     "Aptean Food and Beverage",
    #     "SI Food",
    #     "Indigo WMS",
    #     "UnityF8",
    #     "Exenta ERP",
    #     "Ramsauer & Stürmer",
    #     "Swords",
    #     "SSG Insight",
    #     "Cimdata ERP",
    #     "Objective",
    #     "Proteus",
    #     "Calidus",
    #     "Logis ERP",
    #     "Southware",
    #     "AssetPoint",
    #     "ProcessPro",
    #     "Oxaion ERP",
    #     "Affinitus FreshWare",
    #     "Affinitus GrowMaster",
    #     "Respond",
    #     "Prima Solutions ERP",
    #     "Affinitus ChefServe",
    #     "EDI Direct",
    #     "ImPuls",
    #     "Elucid",
    #     "trend SWM",
    #     "Aptean Retail Planning",
    #     "Master Distribution System",
    #     "Aptean Business Solutions",
    #     "Apparel Business Systems",
    #     "EquipSoft",
    #     "3T Logistics",
    #     "Aptean Retail PLM",
    #     "Global Service",
    #     "Unity",
    #     "Foodware BC",
    #     "Impress",
    #     "irms360",
    #     "Factory",
    #     "TOTALogistix",
    #     "Aptean EAM",
    #     "Encompix",
    #     "Syncos MES",
    #     "LINKFRESH 365 Business Central",
    #     "Drink-IT",
    #     "bc Food",
    #     "Patch OEE",
    #     "Lascom PLM",
    #     "API Pro"
    # ]

    erp_systems = [
        "Made2Manage",
        "Ross",
        "Paragon",
        "Produce Pro ERP",
        "RLM ERP",
        "Apprise",
        "Traverse Global",
        "Full Circle ERP",
        "JustFood",
        "Intuitive"
    ]

    print(f" Product Size - {len(erp_systems)}")
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                process, erp, df_case, df_email, df_post, df_change, df_survey, df_case_kb, df_kb
            )
            for erp in erp_systems
        ]
        
        for future in as_completed(futures):
            try:
                future.result()  # This will raise exceptions if any occurred in threads
            except Exception as e:
                print(f"Error occurred: {e}")