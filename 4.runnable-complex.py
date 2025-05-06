from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0.7, 
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)
# 가상의 Retriever 함수 (실제로는 Vector Store 등에서 검색)
def retrieve_context(query: str) -> str:
    print(f"[Debug] Retrieving context for: {query}")
    # 실제 구현에서는 관련 문서를 찾아서 반환
    if "LCEL" in query:
        return "LCEL stands for LangChain Expression Language. It allows composing chains using the | operator."
    return "No specific context found."

# Retriever를 Runnable로 변환 (간단히 함수를 감싸서)
retriever = RunnableLambda(retrieve_context)

rag_chain_final = (
    # 1단계: Context 검색 및 원본 질문 유지 (병렬 처리)
    # promps에 포함될 내용 생성
    RunnableParallel(
        context=retriever,
        question=RunnablePassthrough() # 입력의 question만 통과시키도록 수정
        # question = lambda x: x["question"] # 이렇게 명시적으로 지정 가능
    )
    # 2단계: 검색된 context와 원본 question으로 프롬프트 생성
    | ChatPromptTemplate.from_messages([
          ("system", "Answer the question based only on the following context:\n{context}"),
          ("human", "{question}") # Passthrough된 딕셔너리에서 question 키 값 사용
      ])
    # 3단계: 모델 호출
    | chat_model
    # 4단계: 결과 파싱 (문자열로)
    | StrOutputParser()
)

print("\n--- Full RAG Chain Execution ---")
final_answer = rag_chain_final.invoke({"question": "What does LCEL stand for?"})
print(final_answer)