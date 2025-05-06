from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

# 0. LLM 모델 준비
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0.7, 
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)


# 1. 벡터 스토어 준비
persist_directory = "./chroma_db_example"
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
# ----------------------------------------------------

# 벡터 스토어에서 Retriever 생성
retriever = vector_store.as_retriever()

# 2. RAG 프롬프트 템플릿 정의
template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

print("RAG 프롬프트 템플릿이 준비되었습니다.")

# 3. 출력 파서 준비 (LLM 응답에서 문자열만 추출)
output_parser = StrOutputParser()

# 4. 검색된 Document 리스트를 단일 문자열로 포맷하는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. RAG Chain 구성 (LCEL 사용)

# RunnableParallel을 사용하여 retriever와 question을 병렬로 처리하고 결과를 딕셔너리로 묶음
# 'context' 키에는 retriever 결과가, 'question' 키에는 원본 질문이 담김
setup_and_retrieval = RunnableParallel(
    {
        "context": retriever | format_docs, 
        "question": RunnablePassthrough()
    }
)

# 위에서 준비된 딕셔너리를 prompt, llm, output_parser로 순차적으로 연결
rag_chain = setup_and_retrieval | prompt | llm | output_parser

print("RAG Chain이 성공적으로 구성되었습니다!")

# RAG Chain 실행
question = "What are the main components of LangChain?"
# question = "How does MMR search work?" # 문서 내용에 따라 질문 변경 가능

print(f"\n--- 질문: {question} ---")

# Chain 실행 (invoke 사용)
answer = rag_chain.invoke(question)

print("\n--- 답변 ---")
print(answer)