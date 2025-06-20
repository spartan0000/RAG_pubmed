import openai
from openai import OpenAI
import chromadb
import os
from chromadb.config import Settings
from typing import List
import json
from dotenv import load_dotenv
load_dotenv()

from Bio import Entrez

openai.api_key = os.getenv('OPENAI_API_KEY')
Entrez.email = os.getenv('EMAIL')

chroma_client = client = chromadb.PersistentClient(path = './chroma_db')
collection = chroma_client.get_or_create_collection('pubmed_cache')

openai_client = OpenAI()

#translate free text query into a query formatted for PubMed
def translate_query(user_query:str) -> str:
    system_prompt = """
    You are helping with a pubmed query.  Take the user input and extract the search terms of interest
    and convert it into a PubMed compatible search using MeSH terms as well as any publication dates if 
    the user has input those.  Return ***ONLY valid JSON*** in the reponse.  Do not explain, format or decorate
    the output.  The JSON should be formatted as follows with the key:value pairs
    {mesh_terms: a list of biomedical terms from the query as MeSH terms where possible,
    publication_date: formatted as a date range YYYY-MM-DD for the start and end dates.  Or, if no dates are input, default to January 1, 2022-December 31, 2024
    pubmed_query: a properly formatted PubMed query search string using [MeSH Terms], [All Fields], and [Publication - Date]
    """
    user_prompt = f"""User's natural language query: {user_query}"""
    
    response = openai_client.responses.create(
        model = 'gpt-4o-mini',
        input = [
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role':'user',
                'content':user_prompt,
            }
        ],
        temperature = 0
    )
    parsed = json.loads(response.output[0].content[0].text)
    pubmed_query = parsed['pubmed_query']
    return pubmed_query



#function to get embeddings
def get_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        input = text,
        model = 'text-embedding-3-small',
    )
    return response.data[0].embedding


#get the articles and metadata using the prompt created above
def parse_pub_date(pub_date):
    if 'Year' in pub_date:
        year = pub_date['Year']
        month = pub_date.get('Month', '01')
        day = pub_date.get('Day', '01')
        return f'{year} - {month} - {day}'
    return 'Not Available'

#get pubmed articles
def get_articles(query, n_results = 5):
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
                'Authors':authors_list,
                'Abstract':abstract,
                'Journal':journal,
                'Keywords':keywords,
                'URL':url,
                'Publication Date':pub_date,
                }
        
            results.append(new_result)
         
    return results

#cache the results in our db
def cache_abstracts(results):
    documents = []
    ids = []
    embeddings = []
    metadatas = []
    
    for result in results:
        abstract = result.get('Abstract', '').strip()
        if not abstract:
            continue
        
        try:
            embedding = get_embedding(abstract)
            if not embedding or not isinstance(embedding, list):
                
                print(f'Skipping PMID {result.get("PMID")} due to invalid embedding')
                continue
        except Exception as e:
            print(f'Error embedding PMID {result.get("PMID")} - invalid embedding')
            continue
        documents.append(abstract)
        ids.append(result['PMID'])
        embeddings.append(embedding)
        metadatas.append({
            'title':result.get('Title', ''),
            'journal':result.get('Journal', ''),
            'authors':result.get('Authors', ''),
            'publication date':result.get('Publication Date', ''),
            'keywords':result.get('Keywords', ''),
        })
            
    if documents:   
        collection.add(
            documents = documents,
            ids = ids,
            embeddings = embeddings,
            metadatas = metadatas,
        )
    else:
        print('No valid documents to cache')

#look up query in vector database
#may need to add some function here to return nothing if no documents meet a threshold for similarity

def vector_db_lookup(query_embedding, n_res = 5):
    results = collection.query(query_embeddings = [query_embedding], n_results = n_res)
    if results['documents']:
        return [
            {
                'abstract':doc,
                'metadata':meta,
                'source':'RAG',
            }
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
    return []

#getting the output formatted to feed the LLM
def format_pubmed_articles(pubmed_results):
    pubmed_articles = []
    for i in range(len(pubmed_results)):
        
        article = "\n".join([
            f"""###Article source: Pubmed
            **Title:** {pubmed_results[i].get('Title', '')}
            **Authors:** {pubmed_results[i].get('Authors', '')}
            **Journal:** {pubmed_results[i].get('Journal', "")}
            **Abstract:** {pubmed_results[i]['Abstract']}
            """
        ])
        pubmed_articles.append(article)
        return ''.join(pubmed_articles)
    
#getting the output formatted to feed the LLM
def format_rag_output(rag_results):
    rag_articles = []
    for i in range(len(rag_results)):
        article =  "\n".join([
            f"""Article(Source: {rag_results[i].get('source', 'Unknown')}
            **Title:** {rag_results[i]['metadata'].get('title', '')}
            **Authors:** {rag_results[i]['metadata'].get('authors', '')}
            **Journal:** {rag_results[i]['metadata'].get('journal', '')}
            **Abstract:** {rag_results[i]['abstract']}
            ---"""])
        rag_articles.append(article)
    return ''.join(rag_articles)  

def summarize(rag_pubmed_output):
    system_prompt = """
    You are a helpful assistant tasked with summarizing scientific articles for a scientist or healthcare professional
    Include the source where possible.  You can use specialized medical jargon and assume that the reader understands.
    Summarize each article highlighting the main points, any statistically significant or clinically significant results and the author's main conclusions.
    Then, at the end, provide 3-4 sentences that give a broad overview of the articles together.
    """

    articles = rag_pubmed_output

    
    response = openai_client.responses.create(
        model = 'gpt-4o-mini',
        input = [
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role':'user',
                'content':articles,
            }
        ],
        temperature = 0
    )
    return response.output[0].content[0].text



if __name__ == '__main__':
<<<<<<< HEAD
    user_query = input('Input a query as well as any dates if relevant: ')
=======
    user_query = input('Input a query term: ')
>>>>>>> df1f54b36b6a1f7d817ce006ba158849c6efcda9
    vector_query = get_embedding(user_query)
    pubmed_query = translate_query(user_query)

    rag_results = vector_db_lookup(vector_query)
<<<<<<< HEAD
    pubmed_results = get_articles(pubmed_query, n_results=5)
=======
    pubmed_results = get_articles(pubmed_query)
>>>>>>> df1f54b36b6a1f7d817ce006ba158849c6efcda9

    cache_abstracts(pubmed_results)

    pubmed_output = format_pubmed_articles(pubmed_results)
    rag_output = format_rag_output(rag_results)

    composite_output = pubmed_output + rag_output

    summary = summarize(composite_output)

    print(summary)
