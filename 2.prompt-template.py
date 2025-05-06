from langchain.prompts import PromptTemplate

# 템플릿 정의: {product} 부분에 변수가 들어감
prompt_template = PromptTemplate.from_template(
    "Suggest a good name for a company that makes {product}."
)

# 템플릿에 변수 값 채우기
formatted_prompt = prompt_template.format(product="colorful socks")

print(formatted_prompt)
# 출력 예시: Suggest a good name for a company that makes colorful socks.

# (만약 LLM 인터페이스 모델이 있다면)
# llm = SomeLLM() # LLM 인터페이스 모델 가정
# response = llm.invoke(formatted_prompt)
# print(response)