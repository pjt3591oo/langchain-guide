from langchain_community.document_loaders import WebBaseLoader

# 웹 페이지 로더 생성 (URL 리스트 전달 가능)
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

# 문서 로드
documents = loader.load()

print(f"Loaded {len(documents)} document(s) from web.")
if documents:
    print("--- Web Document ---")
    print("Content Preview:", documents[0].page_content[:200])
    print("Metadata:", documents[0].metadata) # URL이 메타데이터에 포함됨