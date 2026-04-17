# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str #사용자 질문문
    answer: str #세율
    tax_base_equation: str #과세표준 계산 수식
    tax_deduction: str #공제액
    market_ratio: str #공정시장가액비율
    tax_base: str #과세표준 계산   

graph_builder = StateGraph(AgentState)

# %%
# %pip install -qU pypdf langchain-community langchain-text-splitters

# %%
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1500,
#     chunk_overlap = 100,
#     separators=['\n\n', '\n']
# )

# %%
# from langchain_community.document_loaders import TextLoader

# text_path = './documents/real_estate_tax.txt'

# loader = TextLoader(text_path, encoding='utf-8')
# document_list = loader.load_and_split(text_splitter)

# %%
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

# vector_store = Chroma.from_documents(
#     documents=document_list,
#     embedding=embeddings,
#     collection_name = 'real_estate_tax',
#     persist_directory = './real_estate_tax_collection'
# )

vector_store = Chroma(
        collection_name="real_estate_tax",
        persist_directory="./real_estate_tax_collection",
        embedding_function=embeddings
    )

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# %%
query = "일반인이 5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내야 하나요?"

# %%
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

client = Client()
rag_prompt = client.pull_prompt("rlm/rag-prompt")

llm = ChatOpenAI(model='gpt-5.4')
small_llm = ChatOpenAI(model='gpt-5.4-mini')

# %%
tax_base_retrieval_chain = (
    {'context': retriever, 'question': RunnablePassthrough()} 
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

tax_base_equation_prompt = ChatPromptTemplate.from_messages([
    ('system', '사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요. 과세표준 수식만 리턴해주세요'),
    ('human', '{tax_base_equation_information}')
])

tax_base_equation_chain = (
    {'tax_base_equation_information': RunnablePassthrough()}
    | tax_base_equation_prompt
    | llm
    | StrOutputParser()
)

tax_base_chain = {'tax_base_equation_information' : tax_base_retrieval_chain} | tax_base_equation_chain

def get_tax_base_equation(state: AgentState):
    tax_base_equation_question = '주택에 대한 종합부동산세 계산시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요'
    
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)
    
    return {'tax_base_equation': tax_base_equation}


tax_deduction_chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

def get_tax_deduction(state: AgentState):
    tax_deduction_question = '주택에 대한 종합부동산세 계산시 사용하는 공제금액을 알려주세요'
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)
    return {'tax_deduction': tax_deduction}

from langchain_tavily import TavilySearch
from datetime import date

tavily_search_tool = TavilySearch(
    max_results=3,
    topic="general",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    search_depth="advanced",    
)

tax_market_ratio_prompt = ChatPromptTemplate.from_messages([
('system', f'아래 정보를 기반으로 해당하는 공정시장 가액비율을 계산해 주세요\n\nContext:\n{{context}}'),
('human', '{query}')
])

def get_market_ratio(state: AgentState):
    query = f"오늘 날짜:({date.today()})의 주택에 대한 종합부동산세 계산시 공정시장가액비율은 몇 %인가요?"
    context = tavily_search_tool.invoke(query)
    print(f'context = {context}')
    tax_market_ratio_chain = (
        tax_market_ratio_prompt
        | llm
        | StrOutputParser()
    )
    market_ratio = tax_market_ratio_chain.invoke({'context': context, 'query': query})
    return {'market_ratio': market_ratio}


tax_base_calculation_prompt = ChatPromptTemplate.from_messages(
    [
        ('system',"""
주어진 내용을 기반으로 과세표준을 계산해주세요.

과세표준 계산 공식: {tax_base_equation}
공제금액: {tax_deduction}
공정시장가액비율: {market_ratio}"""),
        ('human', '사용자 주택 공시가격 정보: {query}')
    ]
)

def calculate_tax_base(state: AgentState):
    tax_base_equation = state['tax_base_equation']
    tax_deduction = state['tax_deduction']
    market_ratio = state['market_ratio']
    query = state['query']
    tax_base_calculation_chain =(
        tax_base_calculation_prompt
        | llm
        | StrOutputParser()
    )
    tax_base = tax_base_calculation_chain.invoke({
        'tax_base_equation': tax_base_equation,
        'tax_deduction': tax_deduction,
        'market_ratio': market_ratio,
        'query': query
    })
    return {'tax_base': tax_base}

# %%
# initial_state = {
#     'query': query,
#     'tax_base_equation': '과세표준 = max(0, (주택 공시가격 합산액 - 공제금액) × 공정시장가액비율)',
#     'tax_deduction': '주택에 대한 종합부동산세 계산 시 공제금액은 **1세대 1주택자 12억원**, **일반 개인 9억원**, **법인 또는 법인으로 보는 단체 6억원**입니다.  \n즉, 납세의무자별로 주택 공시가격 합산액에서 해당 공제금액을 뺀 뒤 과세표준을 계산합니다.',
#     'market_ratio': '2026-04-16 기준, **주택에 대한 종합부동산세 공정시장가액비율은 60%**입니다.'
# }

# %%
# calculate_tax_base(initial_state)

# %%
tax_rate_calculate_prompt = ChatPromptTemplate.from_messages([
    ('system', '''당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해 주세요
    종합부동산세 세율: {context}'''),
    ('human', '''과세표준과 사용자가 소지한 주택의 수가 아래와 같을 때 종합부동산세를 계산해주세요
    과세표준: {tax_base}
    주택 수: {query}''')
])

def calculate_tax_rate(state: AgentState):
    query = state['query']
    tax_base = state['tax_base']
    context = retriever.invoke(query)
    tax_rate_chain = (
        tax_rate_calculate_prompt
        | llm
        | StrOutputParser()
    )
    tax_rate = tax_rate_chain.invoke({'query': query, 'tax_base': tax_base, 'context': context})
    print(f'tax_rate == {tax_rate}')
    return {'answer': tax_rate}

# %%
# calculate_tax_base(initial_state)

# %%
# tax_base_state = {'tax_base': '주어진 정보로는 정확한 **세액**까지는 계산할 수 없고, 요청하신 공식에 따라 **과세표준**을 계산할 수 있습니다.\n\n- 납세의무자: **일반 개인**\n- 보유 주택 공시가격 합산액: **5억 + 10억 + 20억 = 35억**\n- 공제금액: **9억**\n- 공정시장가액비율: **60%**\n\n공식:\n- **과세표준 = max(0, (주택 공시가격 합산액 - 공제금액) × 공정시장가액비율)**\n\n계산:\n- **(35억 - 9억) × 60%**\n- **26억 × 0.6 = 15.6억**\n\n따라서 **과세표준은 15억 6천만원**입니다.\n\n참고:\n- 이것은 **세금 자체가 아니라 세금을 매기기 위한 기준 금액**입니다.\n- 실제 **종합부동산세액**을 계산하려면 **세율** 정보가 추가로 필요합니다.', 'query': query}

# %%
# calculate_tax_rate(tax_base_state)

# %%
graph_builder.add_node('get_tax_base_equation', get_tax_base_equation)
graph_builder.add_node('get_tax_deduction', get_tax_deduction)
graph_builder.add_node('get_market_ratio', get_market_ratio)
graph_builder.add_node('calculate_tax_base', calculate_tax_base)
graph_builder.add_node('calculate_tax_rate', calculate_tax_rate)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'get_tax_base_equation')
graph_builder.add_edge(START, 'get_tax_deduction')
graph_builder.add_edge(START, 'get_market_ratio')
graph_builder.add_edge('get_tax_base_equation', 'calculate_tax_base')
graph_builder.add_edge('get_tax_deduction', 'calculate_tax_base')
graph_builder.add_edge('get_market_ratio', 'calculate_tax_base')
graph_builder.add_edge('calculate_tax_base', 'calculate_tax_rate')
graph_builder.add_edge('calculate_tax_rate', END)

# %%
graph = graph_builder.compile()
