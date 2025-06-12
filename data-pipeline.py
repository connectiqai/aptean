import pandas as pd
import pymysql
import re
import logging
from fuzzywuzzy import fuzz
import json

from openai import OpenAI 
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'india.skoruz.com',
    'user': 'chupsdev02',
    'password': 'Chupsdev20$',
    'database': 'aptean_data',
    'cursorclass': pymysql.cursors.DictCursor
}

instructions = "You are an expert in analyzing enterpise softwate support cases."

tools = [
    {
        "type": "function",
        "name": "get_support_case_information",
        "description": (
            "Analyze the support case details (subject, description, email conversation, and related posts) "
            "and extract the root cause, resolution summary, and one high-level category."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "root_cause_summary": {
                    "type": "string",
                    "description": (
                        "Clearly identify the technical, data, process, or user-related root cause based on the complete case information. "
                        "Avoid vague or unsupported statements."
                    )
                },
                "resolution_summary": {
                    "type": "string",
                    "description": (
                        "Describe how the issue was resolved, including key steps, tools, environments, fixes, or validations used."
                    )
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Classify the case into exactly one of the following fixed high-level categories based on the cause and resolution:\n\n"
                        "- Automation: Case was or could be resolved by scripting, scheduling, validation logic, or process automation.\n"
                        "- Product Changes: Issue stemmed from a product defect, limitation, or missing feature that required a config/code/patch/enhancement.\n"
                        "- Education: Case resulted from user misunderstanding, incorrect use, or lack of knowledge; resolved through guidance or documentation.\n"
                        "- Process Improvement: Root cause was due to missing validations, process handoffs, team coordination, or business workflow gaps."
                    ),
                    "enum": [
                        "Automation",
                        "Product Changes",
                        "Education",
                        "Process Improvement"
                    ]
                }
            },
            "required": [
                "root_cause_summary",
                "resolution_summary",
                "category"
            ],
            "additionalProperties": False
        }
    }
]


def get_db_connection():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        logger.debug("Database connection established")
        return conn
    except Exception as e:
        logger.error(f"get_db_connection error: {e}")
        return None

def export_ross_case_numbers_to_excel(product_name, file_name):
    try:
        logger.info(f"Product Name - {product_name}")
        
        with get_db_connection() as connection:
            if connection is None:
                logger.error("DB connection failed. Aborting export.")
                return

            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT case_number FROM support_cases
                    WHERE product_line = %s
                    ORDER BY RAND()
                    LIMIT 500
                """, (product_name,))
                ross_cases = cursor.fetchall()

                if not ross_cases:
                    logger.info("No Ross cases found to export.")
                    return

                case_df = pd.DataFrame(ross_cases)
                
                case_df.to_excel(file_name, index=False)
                logger.info(f"Export completed: {file_name}")

    except Exception as e:
        logger.error(f"export_ross_case_numbers_to_excel error: {e}")

def get_ross_case_data_from_excel(file_name):
    try:
        input_df = pd.read_excel(file_name)
        case_numbers = input_df['case_number'].dropna().astype(int).tolist()

        if not case_numbers:
            logger.warning("No case numbers found in Excel.")
            return []

        connection = get_db_connection()
        if connection is None:
            logger.error("DB connection failed.")
            return []

        results = []
        with connection:
            with connection.cursor() as cursor:
                # Fetch support cases
                format_strings = ','.join(['%s'] * len(case_numbers))
                cursor.execute(f"""
                    SELECT case_number, subject, description, component_name, category_name
                    FROM support_cases
                    WHERE case_number IN ({format_strings})
                """, tuple(case_numbers))
                support_cases = {row['case_number']: row for row in cursor.fetchall()}

                # Fetch posts
                cursor.execute(f"""
                    SELECT case_number, body
                    FROM case_posts
                    WHERE case_number IN ({format_strings})
                    ORDER BY created_date
                """, tuple(case_numbers))
                posts = {}
                for row in cursor.fetchall():
                    cn = row['case_number']
                    if row.get('body'):
                        posts.setdefault(cn, []).append({'body': row['body']})

                # Fetch emails
                cursor.execute(f"""
                    SELECT case_number, text_body
                    FROM case_emails
                    WHERE case_number IN ({format_strings})
                    ORDER BY message_date
                """, tuple(case_numbers))
                emails = {}
                for row in cursor.fetchall():
                    cn = row['case_number']
                    if row.get('text_body'):
                        emails.setdefault(cn, []).append({'text_body': row['text_body']})

                # Combine results
                for cn, case in support_cases.items():
                    results.append({
                        "case_number": case['case_number'],
                        "case_subject": case['subject'],
                        "case_desc": case['description'],
                        "component_name":  case['component_name'],
                        "category_name": case['category_name'],
                        "emails": emails.get(cn, []),
                        "posts": posts.get(cn, [])
                    })

        logger.info(f"Retrieved {len(results)} cases from DB.")
        return results

    except Exception as e:
        logger.error(f"get_ross_case_data_from_excel error: {e}")
        return []

def clean_support_email(text):
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
        "Please do not attempt to respond to this message."
    }

    cutoff_markers = {
        "[logo description automatically generated]",
        "from:",
        "this message was created automatically by the mail system",
        "confidentiality notice:",
        "this message contains confidential information and is intended only for the individual named."
    }

    cutoff_patterns = [
        re.compile(r'-{2,}.*original message.*-{2,}', re.IGNORECASE),
        re.compile(r'^[- ]*original message[- ]*$', re.IGNORECASE)
    ]

    patterns = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
        'cid_image': re.compile(r'\[cid:[^\]]*\]'),
        'phone': re.compile(r'\b(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})\b'),
        'underscore_line': re.compile(r'^_+$'),
        'dash_line': re.compile(r'^-+$'),
    }

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()

        # Stop processing if a cutoff marker or pattern is found
        if any(marker in lowered for marker in cutoff_markers) or any(p.search(stripped) for p in cutoff_patterns):
            break

        if not stripped or lowered in {",", "|", "a:"}:
            continue

        if any(p.search(stripped) for p in patterns.values()):
            continue

        if any(keyword in lowered for keyword in skip_keywords):
            continue

        if lowered.startswith(("subject:", "case subject:")):
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)

def preprocessing(file_name):        
    try:
        logger.info(f"File Name - {file_name}")

        case_list = get_ross_case_data_from_excel(file_name)
        logger.info(f"Total cases retrieved: {len(case_list)}")
        if not case_list:
            return

        logger.debug(f"Sample case: {case_list[0]}")

        for case in case_list:
            # Clean email text bodies
            for email in case.get('emails', []):
                cleaned_email = clean_support_email(email.get('text_body', ''))
                email['text_body'] = cleaned_email

            # Filter duplicate emails using fuzzy matching
            filtered_emails = []
            seen_texts = []

            for email in case['emails']:
                text = email.get('text_body', '')
                # Early skip for exact duplicates
                if text in seen_texts:
                    continue
                # Use fuzzy ratio to filter near duplicates
                is_duplicate = any(fuzz.ratio(text, seen) > 80 for seen in seen_texts)
                if not is_duplicate:
                    filtered_emails.append(email)
                    seen_texts.append(text)

            case['emails'] = filtered_emails

            # Combine filtered email conversation
            case['email_conversation'] = '\n\n'.join(email.get('text_body', '') for email in filtered_emails)
            case['case_posts'] = '\n\n'.join(post.get('body', '') for post in case['posts'])

        client = OpenAI()

        results = []
        for case in case_list:
            case_number = case.get('case_number', 'N/A')
            user_content = (
                f"- Case Number: {case.get('case_number', '')}\n"
                f"- Case Subject: {case.get('case_subject', '')}\n"
                f"- Case Description: {case.get('case_desc', '')}\n"
                f"- Case Email Conversation:\n{case.get('email_conversation', '')}\n"
                f"- Case Posts:\n{case.get('case_posts', '')}\n"
            )

            logger.info(f"Processing case: {case_number}")

            response = client.responses.create(
                model="gpt-4.1",
                instructions=instructions,
                input=[{"role": "user", "content": user_content}],
                tools=tools
            )

            llm_output = {}
            for tool_call in response.output:
                if tool_call.name == "get_support_case_information":
                    llm_output = json.loads(tool_call.arguments)
                    logger.info(f"[{case_number}] Parsed successfully.")

                    results.append({
                        "case_number": case_number,
                        "component": case.get('component_name', ''),
                        "category": case.get('category_name', ''),

                        "root_cause_summary": llm_output.get("root_cause_summary", ""),
                        "resolution_summary": llm_output.get("resolution_summary", ""),
                        "insights":llm_output.get("category","")
                    })


            logger.info(f"LLM output: {llm_output}")

        df = pd.DataFrame(results)
        output_file = "support_case_summary.xlsx"
        df.to_excel(output_file, index=False)
        logger.info(f"Results written to Excel: {output_file}")

    except Exception as e:      
        logger.error(f"preprocessing : error: {e}")

    logger.info("Preprocessing complete.")

if __name__ == "__main__":

    product_name = "Ross"
    file_name = f"{product_name.lower()}_case_number_list.xlsx"

    #export_ross_case_numbers_to_excel(product_name, file_name)
    preprocessing(file_name)
   
