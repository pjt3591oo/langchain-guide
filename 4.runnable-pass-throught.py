from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
# from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 가상의 Retriever 함수 (실제로는 Vector Store 등에서 검색)
def retrieve_context(query: str) -> str:
    print(f"[Debug] Retrieving context for: {query}")
    # 실제 구현에서는 관련 문서를 찾아서 반환
    if "LCEL" in query:
        return "LCEL stands for LangChain Expression Language. It allows composing chains using the | operator."
    return "No specific context found."

# Retriever를 Runnable로 변환 (간단히 함수를 감싸서)
retriever = RunnableLambda(retrieve_context)

# RunnableParallel과 Passthrough 사용
# 입력: {"question": "What is LCEL?"}
# 출력: {"context": "LCEL stands for...", "question": "What is LCEL?"}
setup_and_retrieval = RunnableParallel(
    # retriever는 question을 입력받아 context를 생성
    context=retriever,
    # RunnablePassthrough는 입력을 그대로 전달 (여기서는 입력 딕셔너리 전체)
    # 여기서는 question만 필요하므로, itemgetter를 사용하거나 lambda를 쓸 수도 있음
    # 예: question=itemgetter("question") 또는 question=lambda x: x["question"]
    # 가장 간단한 형태는 입력 자체를 통과시키는 것. 후속 단계에서 필요한 키를 사용.
    question=RunnablePassthrough()
)

# RAG 프롬프트 템플릿
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based only on the following context:\n{context}"),
    # RunnableParallel의 출력에서 question을 사용하기 위해 input_data.question 형태로 접근
    # 또는 setup_and_retrieval 출력 후 후처리 단계에서 prompt 입력을 재구성할 수도 있음.
    # 여기서는 간단하게 프롬프트 내에서 직접 접근하는 방식을 보여줍니다.
    ("human", "{question}")
])

# RAG Chain (아직 모델과 파서는 연결 안 함)
# setup_and_retrieval의 출력: {"context": ..., "input_data": {"question": ...}}
# 이 출력이 rag_prompt로 전달되어 포맷팅됨
rag_chain_part1 = setup_and_retrieval | rag_prompt

# 실행 테스트
result_messages = rag_chain_part1.invoke({"question": "What is LCEL?"})
print("\n--- RAG Prompt Messages ---")
print(result_messages.to_messages())
# 출력 예시:
# --- RAG Prompt Messages ---
# [SystemMessage(content='Answer the question based only on the following context:\nLCEL stands for LangChain Expression Language. It allows composing chains using the | operator.'), HumanMessage(content='What is LCEL?')]