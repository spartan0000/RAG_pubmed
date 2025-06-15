from openai import OpenAI
from typing import TypedDict

import os
from dotenv import load_dotenv
from langchain.tools import Tool

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from Bio import Entrez


import json

import logging

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s =- %(message)s',
)

logger = logging.getLogger(__name__)

load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

import argparse 

Entrez.email = os.getenv('EMAIL')

#use gpt 3.5
llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.2)

def parse_pub_date(pub_date):
    if 'Year' in pub_date:
        year = pub_date['Year']
        month = pub_date.get('Month', '01')
        day = pub_date.get('Day', '01')
        return f'{year} - {month} - {day}'
    return 'Not Available'

def get_articles(query, n_results):
    results = []
    
    handle = Entrez.esearch(db = 'pubmed', term = query, retmax = n_results)
    record = Entrez.read(handle)
    id_list = record['IdList']
    
    #for each pmid, get information about the article
    for pmid in id_list:
        handle = Entrez.efetch(db = 'pubmed', id = pmid, retmode = 'xml')
        records = Entrez.read(handle)
        
        #process each article
        for record in records['PubmedArticle']:
            article = record['MedlineCitation']['Article']
            title = article.get('ArticleTitle', 'Title Not Available')
            abstract = ' '.join(article['Abstract']['AbstractText']) if 'Abstract' in article else ''
            authors_list = ', '.join(a.get('ForeName', '') + ' ' + a.get('LastName', '') for a in article.get('AuthorList', [])) or 'Authors Not Available'
            journal = article['Journal'].get('Title', 'Journal Not Available')
            keywords = ', '.join(k['DescriptorName'] for k in record['MedlineCitation'].get('MeshHeadingList', [])) or 'Keyword Not Available'
            pub_date = parse_pub_date(article['Journal']['JournalIssue']['PubDate'])
            url = f"https://www.ncbi.nlm.nih.gov/pubmed/{pmid}"
            
            new_result = {
                'PMID':pmid,
                'Title':title,
                'Abstract':abstract,
                'Journal':journal,
                'Keywords':keywords,
                'URL':url,
                'Publication Date':pub_date,
                }
        
            results.append(new_result)
         
    return results
    
def create_string(result): #need to convert the results above to a long string to input into the LLM

    text = ' '.join(f"{result[i].get('Title', '')} {result[i].get('Abstract', '')} {result[i].get('Keywords', '')} {result[i].get('Journal', '')} {result[i].get('Publication Date', '')}" for i in range(len(result)))

    return text


class State(TypedDict):
    user_input: str
    query: str
    start_date: str
    end_date: str
    num_articles: int
    abstract_text: str
    summary: str
    formatted_summary: str


def format_query(state: State) -> State:
    
    extract_prompt = PromptTemplate(
        input_variables = ['user_input'],
        template = """
        Given the following user input, extract the search query terms of interest, the start date, end date, and the number of articles requested
        Return a **valid JSON** object with the following key:value pairs
        {{
        "query": the search term of interest as a string
        "start_date": a string in the format YYYY-MM-DD
        "end_date": a string in the format YYYY-MM-DD
        "num_articles": an integer
        }}
        
        user_input: {user_input}""",
    )
    user_input = state['user_input'] #the user_input in the state is part of the initial state that is specified

    
    prompt = extract_prompt.format(user_input = user_input) #user_input var for the prompt = user_input from state['user_input']

    response = llm.invoke(prompt)

    print(f'response: {response}')

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        raise ValueError("Could not parse LLM output as JSON")

    #print(parsed['query'])
    return {
        'query': parsed['query'],
        'start_date': parsed['start_date'],
        'end_date': parsed['end_date'],
        'num_articles': parsed['num_articles'],

    }


def get_article_node(state: State) -> State:
    """
    Get the abstract from pubmed and add it to the state

    """
    logger.info('Getting articles from pubmed')

    start = state['start_date']
    end = state['end_date']
    query = state['query']
    
    #print('input state:', state)
    date_range = f'({start}[Date - Publication] : {end}[Date - Publication])'
    
    queries = f'{query}[Title/Abstract] AND {date_range}'

    print(queries)



    #print('queries:', queries)
    articles = get_articles(queries, state['num_articles'])
    abstract_text = create_string(articles)

    logger.info('Articles retrieved')

    return {'abstract_text': abstract_text}

def summarize_node(state: State) -> State:
    """
    Summarize the text which is a compilation of abstracts from pubmed.  
    

    Parameters: state(State): The current state which contains the abstract text to summarize

    Returns: state(State): the updated state which includes a summary

    """
    logger.info('Summarizing articles')

    text = state['abstract_text']

    prompt_text = 'Summarize each of the following abstracts giving a summary of the methods, key findings and any important conclusions.  After the summary of each abstract, please also list the journal and publication date. Leave a blank line after each abstract so that it is easier to read'

    #define prompt template

    prompt = PromptTemplate(
        input_variables = ['text'],
        template = f"{prompt_text}: {text}",
    )

    message = HumanMessage(content = prompt.format(text = text))

    response = llm.invoke([message]).content.strip()
    logger.info('Summarizing complete')

    return {'summary': response}

def format_summary_node(state: State) -> State:
    raw_summary = state['summary']
    prompt_text = """

    Simplify the following summaries for a general audience who may not have any medical background as if you are a speaker giving a TED talk.  Explain the 
    material clearly and concisely without oversimplifying the science.  Avoid medical jargon or technical terms if possible or explain them clearly if needed.
    Use inspiring language and metaphors.
    Here is the summary: {raw_summary}
    """
    logging.info('Formatting summary for general audience')
    
    prompt = PromptTemplate(
        input_variables = ['raw_summary'],
        template = prompt_text,
    )

    message = HumanMessage(content = prompt.format(raw_summary = raw_summary))
    response = llm.invoke([message]).content.strip()

    logging.info('Formatting complete')

    return {'formatted_summary': response}


builder = StateGraph(State)

builder.add_node("format_query", format_query)
builder.add_node("get_abstracts", get_article_node)
builder.add_node("summarize", summarize_node)
builder.add_node("format_summary", format_summary_node)

builder.set_entry_point("format_query")
builder.add_edge("format_query", "get_abstracts")
builder.add_edge("get_abstracts", "summarize")
builder.add_edge("summarize", "format_summary")


graph = builder.compile()



def run_pubmed(initial_state):
    builder = StateGraph(State)

    builder.add_node("format_query", format_query)
    builder.add_node("get_abstracts", get_article_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("format_summary", format_summary_node)


    builder.set_entry_point("format_query")
    builder.add_edge("format_query", "get_abstracts")
    builder.add_edge("get_abstracts", "summarize")
    builder.add_edge("summarize", "format_summary")

    graph = builder.compile()
    result = graph.invoke(initial_state)
    return result



def main():
    user_input = input('Enter your pubmed query including dates if relevant: ')
    initial_state = {'user_input': user_input}
    result = graph.invoke(initial_state)
    print(result)



if __name__ == '__main__':
    main()