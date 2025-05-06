from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 6편에서 생성 및 저장한 벡터 스토어 로드 가정 ---
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

print("Retriever가 생성되었습니다.")

# Retriever 직접 사용해보기 (결과는 Document 객체 리스트)
query = "What is Retrieval-Augmented Generation?"
retrieved_docs = retriever.invoke(query)

print(f"\n--- Retriever 결과 (Query: '{query}') ---")
for i, doc in enumerate(retrieved_docs):
    print(f"--- Document {i+1} ---")
    print(doc.page_content[:100]) # 내용 미리보기
    print(doc.metadata)
    print("-" * 15)