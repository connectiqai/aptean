import os
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field

from PydanticAdaptorOpenRouter import PydanticAdaptorOpenRouter

tag_info = """
Here is some context on the different pieces of information you may be given access to as part of analyzing a support case. 
Each of the tags mentioned below may or may not be present for every case, but together they provide a comprehensive view of the issue, its diagnosis, and resolution.

<TAG INFO>
1. CASE SUMMARY - This may include internal notes, investigation steps, and customer communications. It captures the overall flow of the case, including the reported issue, context, and actions taken by the support team.
2. ROOT CAUSE - A focused explanation of the technical or process-related root cause that led to the issue. This information is typically derived from internal investigation or resolution analysis.
3. RECOMMENDATION - The specific solution or recommendation that was implemented or proposed to resolve the issue. This may include configuration changes, bug fixes, process improvements, or customer guidance.
</TAG INFO>
"""

new_kb_article_prompt_template = """
You are a senior technical support analyst. Given a support case summary, its root cause, and the recommendation provided, extract the following structured information:

- **Title**: A short, one-line sentence that clearly summarizes the support case.
- **Problem**: A clear and detailed explanation of the root cause behind the issue.
- **Solution**: A concise and complete summary of the recommended fix or resolution.

Ensure your responses are specific, factual, and based only on the provided input. Do not generalize or add assumptions.

Here is some information explaining the schema of the case information provided to you
<CASE INFORMATION SCHEMA DESCRIPTION>
{tag_info}
</CASE INFORMATION SCHEMA DESCRIPTION>

Here is the actual case information
<CASE INFORMATION>
{case_info}
</CASE INFORMATION>
"""

kb_tag_info = """
Here is some context on the different pieces of information you may be given access to as part of analyzing a knowledge base articles and case information. 
Each of the tags mentioned below may or may not be present for every article/case, but together they provide a comprehensive understanding of the issue, its root cause, and the implemented resolution.

<TAG INFO>
1. ARTICLE NUMBER - A unique identifier assigned to the knowledge base article. It is used for reference, indexing, and easy retrieval.
2. TITLE - A short, one-line sentence that summarizes the key issue in the support case. This will form the basis of the article title.
3. PROBLEM - A clear explanation of the root cause of the issue. This typically reflects the result of internal investigation and diagnosis.
4. SOLUTION - A specific and actionable summary of the fix or recommendation that resolved the issue. This should be complete enough for others to apply the resolution.
5. CASE SUMMARY - This may include internal notes, investigation steps, and customer communications. It captures the overall flow of the case, including the reported issue, context, and actions taken by the support team.
6. ROOT CAUSE - A focused explanation of the technical or process-related root cause that led to the issue. This information is typically derived from internal investigation or resolution analysis.
7. RECOMMENDATION - The specific solution or recommendation that was implemented or proposed to resolve the issue. This may include configuration changes, bug fixes, process improvements, or customer guidance.
</TAG INFO>
"""

existing_kb_article_prompt_template = """
You are a senior support knowledge engineer responsible for maintaining and improving internal Knowledge Base (KB) content.

You will be provided with:
- A list of existing KB articles, each containing:
  - ARTICLE NUMBER
  - TITLE
  - PROBLEM: A clear explanation of the root cause of the issue
  - SOLUTION: The implemented fix or recommendation

- A list of support case insights, including:
  - Case Summary: Overview of the issue and investigation
  - Root Cause Summary: The technical cause of the issue
  - Recommendation: The solution that resolved the case

Your task is to compare each case recommendation with the existing KB articles and perform the following:

1. If an exact or very close match is found between our recommendation and the KB articles:
    - Return the matching article_number and mention that 'A relevant KB article already existed to address such issues across different support cases raised'
 
2. If no exact match is found, but a specific articles could be improved slightly to include the case's recommendation:
    - Return the 'article_number' that needs to be updated, and specify the exact details that should be updated in this article, including the 'title' of the existing KB article that coule be updated and the 'Solution' part of the existing KB article with revised steps or insights to address such issues
 
3. If none of the existing KB articles are relevant to address such issues and a new KB article needs to be created to holistically address such issues across support cases
    - Return a new entry with following details:
    - article_number: Leave empty
    - title: Recommended KB Title that is short, crisp and to the point to create this new KB article ,
    - problem: What Problem this KB article is going to address
    - solution: Solution recommendation mentions the exact resolution steps or details around how such issues could be addresses by customers themselves in a self service mode
 
Ensure your evaluation is:
- Thorough and technically accurate
- Focused on reusability and completeness

Your output should be a structured list of article candidates (matched or new), each including:
- article_number
- title
- problem
- solution

Here is some information explaining the schema of the existing knowledge base articles and case information provided to you
<SCHEMA DESCRIPTION>
{tag_info}
</SCHEMA DESCRIPTION>

Here is the actual case information
<CASE INFORMATION>
{case_info}
</CASE INFORMATION>

Here is the existing knowledge base article information provided to you
<KNOWLEDEBASE ARTICLE INFORMATION>
{kb_info}
</KNOWLEDEBASE ARTICLE INFORMATION>
"""

adaptor = PydanticAdaptorOpenRouter(
    openai_client=None,
    openai_api_key=os.getenv('OPENROUTER_API_KEY')
)

class KBArticle(BaseModel):
    article_number: Optional[str] = Field(
        default=None,
        description="A unique identifier for the knowledge base article. Leave empty if this is a suggested new article."
    )
    title: str = Field(..., description="A short, one-line title that briefly and clearly summarizes the support case.")
    problem: str = Field(..., description="A detailed explanation of the root cause of the issue described in the support case.")
    solution: str = Field(..., description="A clear and complete summary of the recommendation or fix that resolves the problem.")

def get_case_info(row: dict) -> str:
    case_info_str = ""
    col_name_tag_list = [
        ('case_summary', 'CASE SUMMARY'),
        ('case_root_cause', 'ROOT CAUSE'),
        ('case_recommendation', 'RECOMMENDATION')
    ]
    for col_name, col_tag in col_name_tag_list:
        val = row.get(col_name)
        if val and not pd.isna(val):
            case_info_str += f"<{col_tag}>\n{val}\n</{col_tag}>\n\n"

    return case_info_str

def get_kb_article_info(row: dict) -> str:
    kb_info_str = ""
    col_name_tag_list = [
        {'article_number', 'ARTICLE NUMBER'},
        ('title', 'TITLE'),
        ('problem', 'PROBLEM'),
        ('resolution', 'SOLUTION')
    ]
    for col_name, col_tag in col_name_tag_list:
        val = row.get(col_name)
        if val and not pd.isna(val):
            kb_info_str += f"<{col_tag}>\n{val}\n</{col_tag}>\n\n"

    return kb_info_str

def kb_analysis(case_summary_dict: dict, kb_dicts: List[dict] = None):
    kb_article_prompt = ""
    if kb_dicts:
        case_info = get_case_info(case_summary_dict)

        kb_info_list = []
        for kb_dict in kb_dicts:
            kb_info_list.append( get_kb_article_info(kb_dict))
        kb_info = "\n\n".join(kb_info_list)

        kb_article_prompt = existing_kb_article_prompt_template.format(
            case_info=case_info,
            tag_info=kb_tag_info,
            kb_info=kb_info
        )
    else:
        case_info = get_case_info(case_summary_dict)

        kb_article_prompt = new_kb_article_prompt_template.format(
            case_info=case_info,
            tag_info=tag_info
        )

    content = [{"type": "text", "text": kb_article_prompt}]
    curr_message = {
        "role": "user",
        "content": content
    }

    message_history = [curr_message]

    kb_article_info = adaptor.chat.completions.create(
        pydantic_model=KBArticle,
        num_retries=5,
        model="gpt-4o-mini",
        messages=message_history,
        max_tokens=4096,
        stream=False
    )

    kb_article_info_dict = kb_article_info.model_dump()
    return kb_article_info_dict

if __name__ == "__main__":
    case_summary_dict = {
        "case_summary": "Multiple customers reported issues related to receiving inventory items that have past expiration dates. They faced system constraints when trying to complete transactions for items that were previously set in the system. Support investigated and determined that by setting specific parameters in the Product Master, such as allowing negative shelf life days and other associated configurations, it enabled the prompt for past expiration dates during the receipt of products. The resolution involved confirming the setup in both Zeus and Hera systems, ensuring the products could be received with dates in the past when properly configured. Testing confirmed successful transactions followed the changes made.",
        "case_root_cause": "The core issue lies in the configuration settings within the Product Master module, specifically around shelf life settings and the ability to override expiration date constraints during receipt processes, which restricted customer workflow in receiving items with previous expiration dates.",
        "case_recommendation": "To address and prevent this issue, the Product Master configuration should provide clearer guidance for users on how to allow the receiving of products with past expiration dates. Specifically, implement a system that allows administrators to set an option for 'Allow receiving of expired products' which can be easily enabled or disabled. Additional documentation or in-system prompts should be created to inform users about necessary settings and potential warnings associated with receiving products with past expiration dates to ensure they can conduct their transactions without interruption."
    }

    kb_analysis(case_summary_dict)
