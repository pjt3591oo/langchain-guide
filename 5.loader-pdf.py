from langchain_community.document_loaders import PyPDFLoader

# PDF 파일 로더 생성
loader = PyPDFLoader("./data/example.pdf")

# 문서 로드 (PDF 페이지별로 Document 객체가 생성될 수 있음)
pages = loader.load_and_split() # load() 또는 load_and_split() 사용 가능

print(f"Loaded {len(pages)} page(s) from PDF.")
if pages:
    print("--- First Page Document ---")
    print("Content Preview:", pages[0].page_content[:-1])
    print("Metadata:", pages[0].metadata) # 페이지 번호 등이 메타데이터에 포함됨