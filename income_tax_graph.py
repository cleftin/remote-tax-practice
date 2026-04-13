# %%
# %pip install -q python-dotenv langchain-openai

# %%
# %pip install -q langgraph

# %%
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="income_tax_collection",
    persist_directory="./income_tax_collection",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={'k': 3})


# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

graph_builder = StateGraph(AgentState)

# %%
def retrieve(state: AgentState):
    query = state['query']
    docs = retriever.invoke(query)
    return {"context": docs}

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o')

# %%
from langsmith import Client

client = Client()
generate_prompt = client.pull_prompt("rlm/rag-prompt")

generate_llm = ChatOpenAI(model='gpt-4o', max_completion_tokens=100)

def generate(state: AgentState):
    context = state['context']
    query = state['query']
    rag_chain = generate_prompt | generate_llm
    response =rag_chain.invoke({'question': query, 'context': context})
    return {'answer': response.content}

# %%
from langsmith import Client
from typing import Literal

client = Client()
doc_relevance_prompt = client.pull_prompt("langchain-ai/rag-document-relevance")

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
    query = state["query"]
    context = state["context"]
    print(f'context == {context}')
    doc_relevance_chain = doc_relevance_prompt | llm
    response = doc_relevance_chain.invoke({'question': query, 'documents': context})
    print(f'doc relevance response: {response}')
    print(type(response))
    if response['Score'] == 1:
        return 'relevant'
    return 'irrelevant'


# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사람과 관련된 표현 -> 거주자']

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요
사전: {dictionary}
질문: {{query}}
""")

def rewrite(state: AgentState):
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    response = rewrite_chain.invoke({'query': query})
    return {'query': response}


# %%
# from langsmith import Client

# client = Client()
# hallucination_prompt = client.pull_prompt("langchain-ai/rag-answer-hallucination")
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

hallucination_prompt = PromptTemplate.from_template(f"""
You are a teacher with evaluating whether a student's answer is based on documnets or not,
Given documents, which are excerpts from income tax law, and a student's answer;
If the student's answer is based on documents, respond with 'not hallucinated',
If the student's answer is not based on documnets, respond with 'hallucinated'.env

doucmnets: {{documents}}
student_answer: {{student_answer}}
""")

hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)

def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']
    context = state['context']
    context = [doc.page_content for doc in context]
    print(f'context == {context}')
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke({'student_answer': answer, 'documents': context})
    print(f'doc relevance response: {response}')
    return response
    # if response['Score'] == 1:
    #     return 'generate'
    # return 'end'

# %%
from langsmith import Client

client = Client()
helpfulness_prompt = client.pull_prompt("langchain-ai/rag-answer-helpfulness")

def check_helpfulness_grader(state: AgentState):
    query = state['query']
    answer = state['answer']
    
    helpfulness_chain = helpfulness_prompt | llm
    response = helpfulness_chain.invoke({'question': query, 'student_answer': answer})
    print(f'helpfulness response: {response}')
    if response['Score'] == 1:
        return 'helpful'
    return 'unhelpful'

def check_helpfulness(state: AgentState):
    return state

# %%
# query = '연봉 5천만원인 거주자의 소득세는 얼마인가요?'
# context = retriever.invoke(query)
# print('document')
# for document in context:
#     print(document.page_content)
# print('document')

# generate_state = {'query': query, 'context': context}
# answer = generate(generate_state)
# print(f'answer == {answer}')

# helpfulness_state ={'query': query, 'answer': answer}
# check_helpfulness(helpfulness_state)

# hallucination_state = {'answer': answer, 'context': context}
# check_hallucination(hallucination_state)

# %%
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
# graph_builder.add_node('check_doc_relevance', check_doc_relevance)
graph_builder.add_node('rewrite', rewrite)
# graph_builder.add_node('check_hallucination', check_hallucination)
graph_builder.add_node('check_helpfulness', check_helpfulness)


# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)
graph_builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'not hallucinated': 'check_helpfulness',
        'hallucinated': 'generate'
    }
)
graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)
graph_builder.add_edge('rewrite', 'retrieve')

# %%
graph = graph_builder.compile()
