from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from typing import List

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0.7, 
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)

# 1. 원하는 출력 형식을 Pydantic 모델로 정의
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

# 2. Output Parser 생성 (Pydantic 모델 지정)
pydantic_parser = PydanticOutputParser(pydantic_object=Joke)

# 3. Parser의 형식 지정 지침 가져오기
format_instructions = pydantic_parser.get_format_instructions()
print(format_instructions)
# 출력 예시: (JSON 스키마와 함께 출력 지침이 나옴)
# The output should be formatted as a JSON instance that conforms to the JSON schema below.
# As an example, for the schema {"title": "foo", "description": "bar"}
# the object {"title": "foo", "description": "bar"} is a well-formatted instance of the schema.
# The object {"title": "foo"} is not well-formatted.
#
# Here is the output schema:
# ```json
# {"properties": {"setup": {"title": "Setup", "description": "question to set up a joke", "type": "string"}, "punchline": {"title": "Punchline", "description": "answer to resolve the joke", "type": "string"}}, "required": ["setup", "punchline"]}
# ```

# 4. 프롬프트 템플릿에 지침 포함
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions}
)

# 5. 모델 호출 (프롬프트 포맷팅 포함)
query = "Tell me a joke."
formatted_prompt = prompt.format(query=query)
ai_response = chat_model.invoke(formatted_prompt)
output = ai_response.content
print("모델 응답 (JSON 문자열):", output)
# 출력 예시: 모델 응답 (JSON 문자열): ```json
# {
#	"setup": "Why don't scientists trust atoms?",
#	"punchline": "Because they make up everything!"
# }
# ```

# 6. Parser로 결과 파싱 (JSON 문자열 -> Pydantic 객체)
try:
    parsed_joke = pydantic_parser.parse(output)
    print("파싱된 결과 (Pydantic 객체):", parsed_joke)
    # 출력 예시: 파싱된 결과 (Pydantic 객체): setup="Why don't scientists trust atoms?" punchline="Because they make up everything!"
    print("농담 설정:", parsed_joke.setup)
except Exception as e:
    print("파싱 오류:", e)
