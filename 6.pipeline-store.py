import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. 문서 로드 (Load) ---
file_path = "./data/example.pdf" # 예시 PDF 파일 경로
loader = PyPDFLoader(file_path)
documents = loader.load()
print(f"'{file_path}'에서 {len(documents)} 페이지를 로드했습니다.")

# --- 2. 문서 분할 (Split) ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)
print(f"{len(documents)}개 페이지를 {len(split_docs)}개의 청크로 분할했습니다.")

# --- 3. 임베딩 모델 준비 (Embed) ---
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print(f"임베딩 모델({model_name})을 준비했습니다.")

# --- 4. 벡터 스토어에 저장 (Store) ---
persist_directory = "./chroma_db_example"
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings_model,
    persist_directory=persist_directory
)
print(f"분할된 문서를 임베딩하여 '{persist_directory}'에 벡터 스토어로 저장했습니다.")

# (선택 사항) 벡터 스토어 강제 저장
vector_store.persist()