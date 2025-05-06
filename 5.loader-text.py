from langchain_community.document_loaders import TextLoader

# 텍스트 파일 로더 생성 (파일 경로 지정)
loader = TextLoader("./data/my_document.txt", encoding='utf-8') # 인코딩 주의

# 문서 로드
documents = loader.load()

# 로드된 문서 확인 (리스트 형태)
print(f"Loaded {len(documents)} document(s).")
if documents:
    print("--- First Document ---")
    print("Content Preview:", documents[0].page_content[:100]) # 내용 미리보기
    print("Metadata:", documents[0].metadata)