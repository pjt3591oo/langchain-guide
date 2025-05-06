# 필요한 라이브러리 설치
# pip install chromadb

from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import FAISS # FAISS 사용 시

# --- 이전 단계에서 준비된 가정 ---
# documents = [...] # splitter.split_documents() 로 얻은 Document 객체 리스트
# embeddings_model = HuggingFaceEmbeddings(...) # 또는 OpenAIEmbeddings() 등
# ---------------------------------

# Chroma 벡터 스토어 생성 및 데이터 저장
# Document 객체 리스트와 임베딩 모델을 전달하여 생성
# persist_directory를 지정하면 해당 경로에 데이터 저장
vector_store = Chroma.from_documents(
    documents=split_docs, # 분할된 Document 객체 리스트
    embedding=embeddings_model, # 사용할 임베딩 모델
    persist_directory="./chroma_db" # 저장될 디렉토리
)

# # FAISS 벡터 스토어 생성 예시 (인메모리)
# vector_store_faiss = FAISS.from_documents(split_docs, embeddings_model)
# # FAISS 로컬 저장/로드
# # vector_store_faiss.save_local("faiss_index")
# # loaded_vector_store = FAISS.load_local("faiss_index", embeddings_model)

print("벡터 스토어 생성이 완료되었습니다.")

# 저장된 벡터 스토어 로드 (애플리케이션 재시작 시 유용)
# loaded_vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)