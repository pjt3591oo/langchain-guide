# 필요한 라이브러리 설치
# pip install langchain-openai

import os
from langchain_openai import OpenAIEmbeddings

# 환경 변수에서 OpenAI API 키 설정 (미리 설정 필요)
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# OpenAI 임베딩 모델 초기화
embeddings_model = OpenAIEmbeddings() # model="text-embedding-ada-002" 등 지정 가능

# 텍스트 임베딩 예시
text = "안녕하세요, LangChain 임베딩 테스트입니다."
embedded_text = embeddings_model.embed_query(text)
print(f"'{text}'의 임베딩 벡터 (일부):", embedded_text[:5]) # 매우 긴 벡터

# 여러 문서 임베딩 예시 (Document 객체 리스트 필요)
# documents = [...] # loader.load() 또는 splitter.split_documents() 결과
# embedded_docs = embeddings_model.embed_documents([doc.page_content for doc in documents])
# print(f"\n{len(documents)}개 문서 임베딩 완료. 첫 번째 문서 벡터 (일부):", embedded_docs[0][:5])