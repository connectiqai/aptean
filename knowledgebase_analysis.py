import os
import pandas as pd
from typing import List, Optional

from pydantic import BaseModel, Field
import tiktoken

from PydanticAdaptorOpenRouter import PydanticAdaptorOpenRouter

tag_info = """
Here is some context on the different pieces of information you may be given access to as part of analyzing a support case. 
Each of the tags mentioned below may or may not be present for every case, but together they provide a comprehensive view of the issue, its diagnosis, and resolution.

<TAG INFO>
1. SUBJECT - This is the initial title or summary provided by the **customer** when raising the case. It usually offers a brief indication of the reported issue and helps establish the **high-level context** of the case.
2. DESCRIPTION - This represents the description of the case and some notes around it, written by the internal support team while discussing the case amongst themselves as they work to resolve it. These notes are typically internal-facing and may contain technical jargon or references to known issues or configurations.
3. POSTS - This represents ad-hoc, often unstructured information present in internal logs, chat notes, or tracking documents as the support team investigates the issue. These entries may contain useful insights, but also a fair amount of noise, such as meeting schedules, team coordination details, or status updates that are not directly relevant to the technical root cause.
4. EMAILS - This section contains the chronological conversation between the **support agent(s)** and the **customer**. It reflects the full communication trail starting from the customer's first report of the issue through to resolution or closure.
   These communications are often informal, and may include misspellings, mixed technical and non-technical language, and emotional tone (e.g., customer frustration or urgency). They are invaluable for understanding the **true impact of the issue**, the **steps taken**, and **what actually resolved it**.
</TAG INFO>
"""

new_kb_article_prompt_template = """
You are a senior technical support analyst. Given the support cases and their details, your task is to draft a knowledgebase article that can help users addresses the issues or queries across these support cases that they have raised in a self help manner. Do not assume anything and the knowledgebase article should content facts only on the basis of the information present in the cases data provided. Return a knowledgebase article in the following format:
 
- **Title**: A short, one-line header title for the knowledgebase title.
- **Problem**: A crisp and clear explanation of the root cause behind the issues that this knowledgebase article is going to address
- **Solution**: A concise and complete summary of the recommended fix or resolution that the users could follow to address such issues themselves
 
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

existing_kb_tag_info = """
Here is some context on the different pieces of information you may be given access to as part of analyzing a knowledge base articles and case information. 
Each of the tags mentioned below may or may not be present for every article/case, but together they provide a comprehensive understanding of the issue, its root cause, and the implemented resolution.

<EXISTING KB TAG INFO>
1. ARTICLE NUMBER - A unique identifier assigned to the knowledge base article. It is used for reference, indexing, and easy retrieval.
2. TITLE - A short, one-line sentence that summarizes the key issue in the support case. This will form the basis of the article title.
3. PROBLEM - A clear explanation of the root cause of the issue. This typically reflects the result of internal investigation and diagnosis.
</EXISTING KB TAG INFO>

<CASE INFO>
1. SUBJECT - This is the initial title or summary provided by the **customer** when raising the case. It usually offers a brief indication of the reported issue and helps establish the **high-level context** of the case.
2. DESCRIPTION - This represents the description of the case and some notes around it, written by the internal support team while discussing the case amongst themselves as they work to resolve it. These notes are typically internal-facing and may contain technical jargon or references to known issues or configurations.
3. POSTS - This represents ad-hoc, often unstructured information present in internal logs, chat notes, or tracking documents as the support team investigates the issue. These entries may contain useful insights, but also a fair amount of noise, such as meeting schedules, team coordination details, or status updates that are not directly relevant to the technical root cause.
4. EMAILS - This section contains the chronological conversation between the **support agent(s)** and the **customer**. It reflects the full communication trail starting from the customer's first report of the issue through to resolution or closure.
   These communications are often informal, and may include misspellings, mixed technical and non-technical language, and emotional tone (e.g., customer frustration or urgency). They are invaluable for understanding the **true impact of the issue**, the **steps taken**, and **what actually resolved it**.
</CASE INFO>
"""

existing_kb_article_prompt_template = """
You are a senior support knowledge engineer responsible for maintaining and improving  Knowledge Base (KB) content.

Your task is to check if the existing knowledgebase articles can address the issues or queries mentioned across these support cases:

1. If an exact or very close matching KB article is found that could holistically address the issues or queries across support cases:
    - Return the matching article_number along with title, problem and solution
 
2. If no exact matching knowledgebase article is found that could fully address the issues or queries across support cases, but if it could be enhances slightly or enriched with some more relevant information to address the issues or queries:
    - Return the article_number that needs to be updated, including the title, problem and solution part of the existing KB article with revised steps or insights to address such issues

3. If none of the existing knowledgebase articles are relevant to address such issues or queries across support cases and a new knowledgebase article needs to be created to holistically address such issues or queries across support cases:
    - Return a new entry with following details:
    - article_number: Leave empty
    - title: Recommended knowledgebase Title that is short, crisp and to the point to create this new knowledgebase article
    - problem: What Problem this knowledgebase article is going to address
    - solution: knowledgebase content explaining the exact resolution steps or details around how such issues could be addresses by customers themselves in a self service mode
 
Ensure your knowledgebase enhancement or new knowledgebase creation is:
- Thorough and only on the basis of input cases data provided without any assumptions
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
    article_number: Optional[int] = Field(
        default=None,
        description="A unique numeric identifier for the knowledge base article. Leave empty (null) if this is a suggested new article."
    )
    title: str = Field(..., description="A short, one-line title that briefly and clearly summarizes the support case.")
    problem: str = Field(..., description="A detailed explanation of the root cause of the issue described in the support case.")
    solution: str = Field(..., description="A clear and complete summary of the recommendation or fix that resolves the problem.")

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

def trim_to_3500_tokens(text: str, model: str = "gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    trimmed_tokens = tokens[:3500]
    return encoding.decode(trimmed_tokens)

def kb_analysis(case_summary_dict: dict, kb_dicts: List[dict] = None):
    kb_article_prompt = ""
    if kb_dicts:
        case_info = case_summary_dict.get('cluster_cases_str','')
        case_info = trim_to_3500_tokens(case_info)

        kb_info_list = []
        for kb_dict in kb_dicts:
            kb_info_list.append( get_kb_article_info(kb_dict))
        kb_info = "\n\n".join(kb_info_list)

        kb_article_prompt = existing_kb_article_prompt_template.format(
            case_info=case_info,
            tag_info=existing_kb_tag_info,
            kb_info=kb_info
        )
    else:
        case_info = case_summary_dict.get('cluster_cases_str','')

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
