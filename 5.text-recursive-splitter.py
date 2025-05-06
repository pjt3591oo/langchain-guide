from langchain.text_splitter import RecursiveCharacterTextSplitter

long_text = """
LangChain은 LLM을 활용한 애플리케이션 개발을 돕는 프레임워크입니다.
다양한 컴포넌트(Models, Prompts, Chains, Agents, Memory, Retrievers 등)를 제공하여
개발자가 복잡한 워크플로우를 쉽게 구축할 수 있도록 지원합니다.
특히 RAG(Retrieval-Augmented Generation)는 외부 데이터를 활용하여 LLM의 한계를 극복하는 강력한 기능입니다.
이번 시간에는 문서 로딩과 텍스트 분할에 대해 배우고 있습니다. 다음 시간에는 임베딩과 벡터 스토어를 다룰 예정입니다.
"""

# RecursiveCharacterTextSplitter 생성
recursive_splitter = RecursiveCharacterTextSplitter(
    # separators=["\n\n", "\n", " ", ""], # 기본값, 필요시 커스텀 가능
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

# 텍스트 분할
# 실제 사용 시: chunks = recursive_splitter.split_documents(documents)
recursive_chunks = recursive_splitter.split_text(long_text)

print(f"Split into {len(recursive_chunks)} chunks using Recursive splitter.")
for i, chunk in enumerate(recursive_chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print("-" * 20)