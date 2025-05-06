from langchain.text_splitter import CharacterTextSplitter

# 예시 텍스트 (실제로는 loader.load()로 얻은 documents 리스트 사용)
long_text = """
LangChain은 LLM을 활용한 애플리케이션 개발을 돕는 프레임워크입니다.
다양한 컴포넌트(Models, Prompts, Chains, Agents, Memory, Retrievers 등)를 제공하여
개발자가 복잡한 워크플로우를 쉽게 구축할 수 있도록 지원합니다.
특히 RAG(Retrieval-Augmented Generation)는 외부 데이터를 활용하여 LLM의 한계를 극복하는 강력한 기능입니다.
이번 시간에는 문서 로딩과 텍스트 분할에 대해 배우고 있습니다. 다음 시간에는 임베딩과 벡터 스토어를 다룰 예정입니다.
"""

# CharacterTextSplitter 생성
text_splitter = CharacterTextSplitter(
    separator="\n\n",  # 문단 기준으로 먼저 나누도록 시도
    chunk_size=100,    # 청크 크기 (문자 수)
    chunk_overlap=20,  # 청크 간 겹침 (문자 수)
    length_function=len, # 길이 계산 함수
)

# 텍스트 분할 (Document 객체 또는 문자열 리스트를 입력으로 받음)
# 실제 사용 시: chunks = text_splitter.split_documents(documents)
chunks = text_splitter.split_text(long_text)

print(f"Split into {len(chunks)} chunks.")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print("-" * 20)