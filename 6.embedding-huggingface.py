# 필요한 라이브러리 설치
# pip install langchain-community sentence-transformers

from langchain_community.embeddings import HuggingFaceEmbeddings

# Hugging Face 임베딩 모델 초기화
# (처음 실행 시 모델 다운로드 시간이 소요될 수 있음)
model_name = "sentence-transformers/all-MiniLM-L6-v2" # 영어권에서 성능 좋은 경량 모델
# model_name = "jhgan/ko-sroberta-multitask" # 한국어 모델 예시
model_kwargs = {'device': 'cpu'} # CPU 사용 명시 (GPU 사용 가능 시 'cuda')
encode_kwargs = {'normalize_embeddings': False} # 정규화 여부

embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 텍스트 임베딩 예시
text = "안녕하세요, LangChain 임베딩 테스트입니다."
embedded_text = embeddings_model.embed_query(text)
print(f"'{text}'의 임베딩 벡터 (일부):", embedded_text[:5])

# 여러 문서 임베딩 예시
# documents = [...]
# embedded_docs = embeddings_model.embed_documents([doc.page_content for doc in documents])
# print(f"\n{len(documents)}개 문서 임베딩 완료. 첫 번째 문서 벡터 (일부):", embedded_docs[0][:5])