from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0.7, 
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)

# 1. Output Parser 생성
output_parser = CommaSeparatedListOutputParser()

# 2. Parser의 형식 지정 지침 가져오기
format_instructions = output_parser.get_format_instructions()
# print(format_instructions)
# 출력 예시: Your response should be a list of comma separated values, eg: `foo, bar, baz`

# 3. 프롬프트 템플릿에 지침 포함
prompt = PromptTemplate(
    template="List 5 ice cream flavors.\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": format_instructions}, # 프롬프트에 지침 주입
)

# 4. 모델 호출 (프롬프트 포맷팅 포함)
formatted_prompt = prompt.format()
print(formatted_prompt)

# 5. 모델 호출
ai_response = chat_model.invoke(formatted_prompt)
print(ai_response)
