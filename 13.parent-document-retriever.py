from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore # 혹은 다른 ByteStore 구현체 사용 가능 (e.g., RedisStore, UpstashRedisStore)
from langchain_community.vectorstores import Chroma # 또는 FAISS 등 다른 벡터 저장소
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()


# 예시: HuggingFace Embeddings 사용 (API 키 불필요)
# model_name = "sentence-transformers/all-MiniLM-L6-v2" # 영어권 경량 모델
model_name = "jhgan/ko-sroberta-multitask" # 한국어 모델 예시
model_kwargs = {'device': 'cpu'} # CPU 사용 명시 (GPU 가능 시 'cuda')
encode_kwargs = {'normalize_embeddings': False} # 정규화 여부 (모델에 따라 다름)
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print(f"Using HuggingFace Embeddings: {model_name}")


# --- 2. 샘플 문서 준비 ---
# 여기서는 간단한 텍스트를 사용하지만, TextLoader 등으로 실제 파일 로드 가능
sample_docs_content = [
    """LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발을 간소화하기 위한 프레임워크입니다.
    이 프레임워크는 데이터 인식(data-aware) 및 에이전트(agentic) 특성을 핵심으로 합니다.
    개발자는 LangChain을 통해 LLM을 외부 데이터 소스와 연결하고, LLM이 특정 환경과 상호작용하도록 할 수 있습니다.
    주요 구성 요소로는 모델 I/O, 검색(Retrieval), 체인(Chains), 에이전트(Agents), 메모리(Memory), 콜백(Callbacks) 등이 있습니다.
    이러한 구성 요소들은 모듈식으로 설계되어 있어, 복잡한 워크플로우도 쉽게 구축할 수 있도록 지원합니다.
    예를 들어, 문서를 로드하고, 텍스트를 분할하고, 임베딩을 생성하고, 벡터 저장소에 저장한 후,
    이 저장된 정보를 바탕으로 질문에 답변하는 RAG(Retrieval Augmented Generation) 시스템을 만들 수 있습니다.
    LangChain은 Python과 JavaScript 라이브러리를 제공하여 다양한 개발 환경을 지원합니다.
    커뮤니티도 활발하여 지속적인 업데이트와 새로운 기능 추가가 이루어지고 있습니다.""",

    """인공지능(AI)은 현대 사회의 다양한 분야에서 혁신을 주도하고 있습니다.
    머신러닝, 딥러닝, 자연어 처리(NLP), 컴퓨터 비전 등 여러 하위 분야로 구성됩니다.
    머신러닝은 데이터로부터 패턴을 학습하여 예측이나 결정을 내리는 기술입니다.
    딥러닝은 인공신경망, 특히 여러 계층으로 구성된 심층 신경망을 사용하여 복잡한 문제를 해결합니다.
    NLP는 인간의 언어를 컴퓨터가 이해하고 처리할 수 있도록 하는 기술로, 챗봇, 번역, 감성 분석 등에 활용됩니다.
    컴퓨터 비전은 시각적 정보를 해석하고 이해하는 기술로, 이미지 인식, 객체 탐지 등에 사용됩니다.
    AI의 발전은 윤리적 고려 사항도 함께 제기하며, 공정성, 투명성, 책임성 확보가 중요한 과제로 남아있습니다.
    또한, AI 기술의 접근성을 높이고 다양한 산업에 적용하기 위한 연구가 활발히 진행 중입니다."""
]

# Document 객체로 변환
original_documents = []
for i, content in enumerate(sample_docs_content):
    original_documents.append(Document(page_content=content, metadata={"source": f"sample_doc_{i+1}", "doc_id": f"doc_{i+1}"}))

# --- 3. 분할기(Splitter) 정의 ---
# 부모 문서를 위한 분할기 (선택 사항, 원본 문서를 그대로 부모로 사용할 수도 있음)
# ParentDocumentRetriever는 add_documents 시점에 parent_splitter를 적용하거나,
# 이미 분할된 부모 문서를 전달받을 수 있습니다.
# 여기서는 원본 문서를 그대로 부모로 사용하고, 자식 청크만 나눕니다.
# parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# 자식 문서를 위한 분할기 (벡터 저장소에 저장될 작은 청크)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)


# --- 4. 저장소(Store) 설정 ---
# Vector Store: 자식 청크의 임베딩을 저장
# persist_directory를 지정하여 디스크에 저장하고 재사용 가능
vectorstore = Chroma(
    collection_name="split_parents_children",
    embedding_function=embeddings_model,
    persist_directory="./chroma_db_parent_retriever" # 저장 경로
)
vectorstore.persist() # DB 변경사항 디스크에 즉시 반영

# Docstore: 원본(부모) 문서를 저장할 저장소 (Key-Value 형태)
# InMemoryStore는 간단한 인메모리 저장소입니다.
# 실제 운영 환경에서는 Redis, SQL 데이터베이스 등 영구적인 저장소 사용을 고려해야 합니다.
docstore = InMemoryStore()


# --- 5. ParentDocumentRetriever 초기화 ---
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    # parent_splitter=parent_splitter, # 원본 문서를 더 큰 부모 청크로 나누고 싶을 때 사용
    # id_key="doc_id", # Document 메타데이터에서 부모 문서 ID로 사용할 키 (기본값: "doc_id")
    # child_metadata_fields=["source"] # 자식 청크에 복사할 부모 메타데이터 필드
)

# --- 6. 문서 추가 ---
# add_documents는 내부적으로 child_splitter를 사용하여 문서를 자식 청크로 나누고,
# 원본 문서는 docstore에, 자식 청크의 임베딩은 vectorstore에 저장합니다.
# ids를 제공하면 해당 ID로 docstore에 저장됩니다. 제공하지 않으면 UUID가 생성됩니다.
doc_ids = [doc.metadata["doc_id"] for doc in original_documents]
retriever.add_documents(original_documents, ids=doc_ids, add_to_docstore=True)

print(f"\n총 {len(original_documents)}개의 원본 문서가 추가되었습니다.")
print(f"Docstore에 저장된 문서 수: {len(list(docstore.yield_keys()))}")
# Chroma의 경우 내부적으로 자식 청크 수를 확인하는 직접적인 방법은 간단치 않으나,
# add_documents가 성공적으로 실행되면 자식 청크들이 vectorstore에 저장된 것입니다.


# --- 7. 검색 수행 ---
query1 = "LangChain의 주요 구성 요소는 무엇인가요?"
print(f"\n--- 질문 1: {query1} ---")
retrieved_docs1 = retriever.get_relevant_documents(query1) # 기본 k=4

if retrieved_docs1:
    print(f"\n[검색된 부모 문서 (질문 1)]:")
    for i, doc in enumerate(retrieved_docs1):
        print(f"--- 문서 {i+1} (출처: {doc.metadata.get('source', 'N/A')}) ---")
        print(doc.page_content[:500] + "...") # 내용 일부만 출력
        print("-" * 20)
else:
    print("검색된 문서가 없습니다.")


query2 = "AI의 윤리적 문제는 무엇인가요?"
print(f"\n--- 질문 2: {query2} ---")
# 검색 시 k 값 (반환할 문서 수) 조절 가능
# ParentDocumentRetriever는 k개의 *자식* 청크를 찾고, 그에 해당하는 *부모* 문서를 반환합니다.
# 따라서 중복된 부모 문서가 반환될 수 있으며, 실제 반환되는 부모 문서 수는 k보다 작거나 같을 수 있습니다.
# search_kwargs를 통해 vectorstore 검색 옵션 조절 가능
retrieved_docs2 = retriever.invoke(query2, config={"configurable": {"search_kwargs": {"k": 2}}})


if retrieved_docs2:
    print(f"\n[검색된 부모 문서 (질문 2)]:")
    for i, doc in enumerate(retrieved_docs2):
        print(f"--- 문서 {i+1} (출처: {doc.metadata.get('source', 'N/A')}) ---")
        print(doc.page_content[:500] + "...") # 내용 일부만 출력
        print("-" * 20)
else:
    print("검색된 문서가 없습니다.")

# --- 8. (선택 사항) 저장소 내용 확인 ---
print("\n--- 저장소 상태 확인 ---")
print("Docstore에 저장된 키 (부모 문서 ID):")
for key in docstore.yield_keys():
    print(f"- {key}")

# Chroma vectorstore에 저장된 자식 청크의 수를 직접 확인하는 것은 API가 복잡할 수 있습니다.
# retriever.vectorstore.delete_collection() # 테스트 후 컬렉션 삭제 원할 시
# import shutil # 디렉토리 삭제
# shutil.rmtree("./chroma_db_parent_retriever") # 테스트 후 DB 디렉토리 삭제 원할 시