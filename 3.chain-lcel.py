from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# 0. 모델 준비
chat_model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0.7, 
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)

# 1. Output Parser 준비
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
pydantic_parser = PydanticOutputParser(pydantic_object=Joke)
format_instructions = pydantic_parser.get_format_instructions()

# 2. 프롬프트 템플릿 준비
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI that tells jokes based on a topic."),
    ("human", "Tell me a joke about {topic}.\n{format_instructions}")
])


# Chain 생성
chain = prompt | chat_model | pydantic_parser

# Chain 실행
topic = "bears"
response_lcel = chain.invoke({"topic": topic, "format_instructions": format_instructions}) # 입력 변수 전달

print(response_lcel) 
print(response_lcel.setup) 
print(response_lcel.punchline) 

