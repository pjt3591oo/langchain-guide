from fastapi import FastAPI
from langserve import add_routes
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader # 예시용 로더
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Hugging Face 임베딩 모델 초기화
# (처음 실행 시 모델 다운로드 시간이 소요될 수 있음)
model_name = "sentence-transformers/all-MiniLM-L6-v2" # 영어권에서 성능 좋은 경량 모델
# model_name = "jhgan/ko-sroberta-multitask" # 한국어 모델 예시
model_kwargs = {'device': 'cpu'} # CPU 사용 명시 (GPU 사용 가능 시 'cuda')
encode_kwargs = {'normalize_embeddings': False} # 정규화 여부


# --- 0. (선택) 환경 변수 설정 ---
# import os
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# --- 1. 배포할 Runnable 생성 (예: 간단 RAG Chain) ---

# 가상의 문서 로드 및 분할 (실제 환경에서는 DB 등에서 가져올 수 있음)
try:
    # 예시 텍스트 파일 생성 (없을 경우)
    with open("./data/my_document.txt", "w", encoding='utf-8') as f:
        f.write("LangServe는 LangChain Runnable을 API로 쉽게 배포하는 라이브러리입니다.\n")
        f.write("FastAPI를 기반으로 하며, 스트리밍과 배치 처리를 지원합니다.\n")
        f.write("LangSmith와 함께 사용하면 개발부터 배포, 모니터링까지 편리하게 관리할 수 있습니다.")

    loader = TextLoader("./data/my_document.txt", encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # 임베딩 및 벡터 스토어 생성
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

except Exception as e:
    print(f"초기 설정 중 오류 발생 (OpenAI 키 확인 또는 파일 경로 확인): {e}")
    # 실제 서비스에서는 더 강력한 오류 처리 필요
    # 여기서는 오류 시 기본적인 LLMChain으로 대체
    retriever = None # Retriever 설정 실패 표시


# # 모델 및 프롬프트 설정
model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0.7, 
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)
prompt = ChatPromptTemplate.from_template("""주어진 내용을 바탕으로 질문에 답해주세요.
내용: {context}
질문: {question}
답변:""")

# RAG Chain 정의
if retriever:
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
else:
    # Retriever 실패 시 간단한 LLMChain으로 대체 (예시)
    prompt_fallback = ChatPromptTemplate.from_template("{question} 에 대해 답해주세요.")
    rag_chain = (
        RunnablePassthrough() # 입력을 그대로 question으로 사용
        | prompt_fallback
        | model
        | StrOutputParser()
    )


# --- 2. FastAPI 앱 생성 ---
app = FastAPI(
    title="LangChain RAG Server",
    version="1.0",
    description="LangServe를 이용한 간단한 RAG API 서버",
)

# --- 3. LangServe를 이용해 Runnable을 API 라우트로 추가 ---
# add_routes 함수가 핵심입니다!
add_routes(
    app,
    rag_chain, # 배포할 Runnable 객체
    path="/rag-chain", # API 엔드포인트 경로 설정
)

# --- 4. (선택) 서버 실행 코드 (개발용) ---
if __name__ == "__main__":
    # 이 파일을 직접 실행할 때 uvicorn 서버를 시작합니다.
    # 실제 배포 환경에서는 gunicorn 등과 함께 사용될 수 있습니다.
    uvicorn.run(app, host="localhost", port=8000)