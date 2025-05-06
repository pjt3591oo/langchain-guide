import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

chat_model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0.7, 
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
pydantic_parser = PydanticOutputParser(pydantic_object=Joke)
format_instructions = pydantic_parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI that tells jokes based on a topic."),
    ("human", "Tell me a joke about {topic}.\n{format_instructions}")
])

chain_lcel = prompt | chat_model | pydantic_parser

# --- batch 예제 ---
print("\n--- Batch Execution Start ---")
topics_to_process = [
    {"topic": "computers", "format_instructions": format_instructions},
    {"topic": "coffee", "format_instructions": format_instructions},
]

# batch 메서드로 여러 입력 동시 처리 (내부적으로 병렬 처리될 수 있음)
batch_results = chain_lcel.batch(topics_to_process)

print(batch_results)
# 출력 예시: [Joke(setup='Why did the computer keep sneezing?', punchline='It had a virus!'), Joke(setup='How does Moses make coffee?', punchline='He brews it.')]
print("Batch 결과 개수:", len(batch_results))
print("첫 번째 농담:", batch_results[0].punchline)
print("--- Batch Execution End ---")


# --- stream 예제 (StrOutputParser 사용) ---
stream_chain = prompt | chat_model | StrOutputParser() # 스트리밍을 위해 문자열 파서 사용

print("\n--- Streaming Execution Start ---")
topic_to_stream = {"topic": "books", "format_instructions": format_instructions}

# stream 메서드로 결과 실시간 받기
full_response = ""
for chunk in stream_chain.stream(topic_to_stream):
    print(chunk, end="", flush=True)
    full_response += chunk # 필요하다면 청크를 모아서 전체 응답 구성 가능

print("\n--- Streaming Execution End ---")
# print("Full streamed response:", full_response) # 전체 응답 확인