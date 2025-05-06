# 필요한 라이브러리 설치 (이전 포스팅에서 설치했다면 생략)
# pip install langchain langchain-openai python-dotenv pydantic
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain # 이전 방식의 Chain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

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

# LLMChain 생성 (Prompt + LLM)
chain_old = LLMChain(llm=chat_model, prompt=prompt)

# Chain 실행
topic = "bears"
formatted_prompt = prompt.format_prompt(topic=topic, format_instructions=format_instructions).to_messages() # 직접 포맷팅 필요
response_old = chain_old.invoke({"topic": topic, "format_instructions": format_instructions}) # 입력 변수 전달

print(response_old) # {'topic': 'bears', 'format_instructions': '...', 'text': '{\n\t"setup": "Why don\'t bears wear shoes?",\n\t"punchline": "Because they have bear feet!"\n}'}

# 결과 파싱은 별도로 필요
try:
    parsed_joke = pydantic_parser.parse(response_old['text'])
    print(parsed_joke)
except Exception as e:
    print("Parsing error:", e)

