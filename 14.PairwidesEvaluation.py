from langchain.evaluation import load_evaluator
from langchain.evaluation import EvaluatorType
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0.7, 
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)

# 예시: 간단한 요약 Chain
summarize_prompt = ChatPromptTemplate.from_template("다음 텍스트를 한 문장으로 요약해줘: {text}")
summarizer_chain = summarize_prompt | llm | StrOutputParser()

example_text = "LangChain은 LLM을 활용한 애플리케이션 개발을 돕는 프레임워크입니다. 다양한 컴포넌트와 체인을 제공하여 복잡한 워크플로우 구축을 간소화합니다."
prediction = summarizer_chain.invoke({"text": example_text})

reference_summary = "LangChain은 LLM 앱 개발을 위한 프레임워크로, 컴포넌트와 체인을 통해 복잡한 워크플로우를 단순화합니다."


# 여러 기준을 동시에 평가할 수도 있습니다.
custom_criteria = {
    "accuracy": "예측이 원본 텍스트의 핵심 정보를 정확하게 반영하는가?",
    "clarity": "예측이 명확하고 이해하기 쉬운가?"
}

evaluator = load_evaluator("labeled_criteria", criteria="correctness", llm=llm)

result = evaluator.evaluate_strings(
    prediction=prediction,
    reference=reference_summary,
    input=example_text
)
print(f"레이블 기반 정확성 평가 결과: {result}")