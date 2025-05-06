from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
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


# 농담 생성 Chain (위에서 정의한 chain_lcel 재사용)
joke_chain = prompt | chat_model | pydantic_parser

# 간단 설명 생성 Chain
explanation_prompt = ChatPromptTemplate.from_template(
    "Briefly explain what {topic} is in one sentence."
)
explanation_chain = explanation_prompt | chat_model | StrOutputParser()

# 두 Chain을 병렬로 실행하는 Combined Chain
# 입력: {"topic": "..."}
# 출력: {"joke": Joke(...), "explanation": "..."}
combined_chain = RunnableParallel(
    joke=joke_chain,
    explanation=explanation_chain,
)

# 실행
topic = "AI"
# RunnableParallel은 각 Runnable에 필요한 입력 키를 자동으로 전달합니다.
# joke_chain과 explanation_chain 모두 'topic' 변수가 필요합니다.
# format_instructions는 joke_chain에만 필요하므로, partial을 사용하거나
# RunnableParallel의 각 항목에 lambda를 써서 입력을 재구성할 수 있습니다.
# 여기서는 joke_chain이 format_instructions를 필요로 하므로, 입력을 조금 수정합니다.

combined_result = combined_chain.invoke({
    "topic": topic,
    "format_instructions": format_instructions # joke_chain을 위해 전달
})

print("\n--- Parallel Execution Result ---")
print("Generated Joke:", combined_result['joke'])
print("Generated Explanation:", combined_result['explanation'])
# 출력 예시:
# --- Parallel Execution Result ---
# Generated Joke: setup='Why did the AI break up with the computer?' punchline="Because it said it needed space!"
# Generated Explanation: AI, or Artificial Intelligence, refers to the simulation of human intelligence processes by machines, especially computer systems.