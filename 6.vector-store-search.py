from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print(f"임베딩 모델({model_name})을 준비했습니다.")

vector_store = Chroma(persist_directory="./chroma_db_example", embedding_function=embeddings_model)
# 사용자 질문
query = "What is Retrieval-Augmented Generation?"

# 벡터 스토어 로드 (이미 생성 및 저장된 경우)
# loaded_vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)

# 유사도 검색 수행
# k: 반환할 결과(청크)의 수
similar_chunks = vector_store.similarity_search(query, k=3)

print(f"\n--- Query: '{query}' 와 유사한 상위 {len(similar_chunks)}개 청크 ---")
for i, chunk in enumerate(similar_chunks):
    print(f"--- Chunk {i+1} (Score: N/A for basic search) ---") # 일부 벡터스토어는 score 반환 가능
    print("Content:", chunk.page_content[:200]) # 내용 미리보기
    print("Metadata:", chunk.metadata)
    print("-" * 20)