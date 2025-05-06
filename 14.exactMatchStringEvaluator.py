from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

from langchain.evaluation import ExactMatchStringEvaluator

load_dotenv()

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0.7, 
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)

# 예시: 간단한 요약 Chain
summarize_prompt = ChatPromptTemplate.from_template("다음 텍스트를 한 문장으로 요약해줘: {text}")
summarizer_chain = summarize_prompt | llm | StrOutputParser()

# 평가할 예시 텍스트 및 예측 결과
example_text = "LangChain은 LLM을 활용한 애플리케이션 개발을 돕는 프레임워크입니다. 다양한 컴포넌트와 체인을 제공하여 복잡한 워크플로우 구축을 간소화합니다."
prediction = summarizer_chain.invoke({"text": example_text})
print(f"예측 요약: {prediction}")

# 사람이 작성한 참조 요약 (선택 사항)
reference_summary = "LangChain은 LLM 앱 개발을 위한 프레임워크로, 컴포넌트와 체인을 통해 복잡한 워크플로우를 단순화합니다."

evaluator = ExactMatchStringEvaluator()
result = evaluator.evaluate_strings(
    prediction=prediction,
    reference=reference_summary,
    input=example_text # 평가지표 계산에 input이 필요할 수 있음 (선택적)
)
print(f"정확 일치 평가 결과: {result}")
# {'score': 0} # 정확히 같지 않으므로 0