from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun # Langchain community의 DuckDuckGo 검색 도구
from langchain.tools import Tool # 사용자 정의 도구 또는 기존 도구를 래핑할 때 사용
from langchain import hub # LangChain Hub에서 프롬프트 등을 가져오기 위함
from langchain.chains import LLMMathChain # LLM 기반 계산기 도구
from dotenv import load_dotenv

load_dotenv()
# --- 1. 환경 변수 설정 (필요시) ---
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY" # 실제 키로 교체하거나 환경 변수 사용

# --- 2. LLM 준비 ---
# 여기서는 OpenAI의 GPT-4 모델을 사용합니다. 다른 모델도 가능합니다.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model_name="l3-8b-stheno-v3.1-iq-imatrix",
    temperature=0.7
)

# --- 3. 도구(Tools) 정의 ---
# DuckDuckGo 검색 도구
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="duckduckgo_search",
    description="최신 정보를 검색해야 할 때 사용합니다. 특정 인물, 사건, 주제에 대한 현재 정보를 얻을 때 유용합니다.",
    func=search.run,
)

# LLM 기반 계산기 도구 (수학 문제 해결용)
# 참고: LLMMathChain은 내부적으로 LLM을 사용하여 수학적 표현식을 파싱하고 계산합니다.
math_chain = LLMMathChain.from_llm(llm=llm, verbose=False) # verbose=True로 하면 내부 동작 확인 가능
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="수학적인 계산이나 연산이 필요할 때 사용합니다. 복잡한 수식이나 단위 변환 등에 유용합니다.",
)

# 사용할 도구 리스트 정의
tools = [search_tool, calculator_tool]

# --- 4. Agent가 사용할 프롬프트 가져오기 ---
# LangChain Hub에서 ReAct 방식의 프롬프트를 가져옵니다.
# 이 프롬프트는 LLM에게 어떻게 생각하고 행동(도구 사용)해야 하는지 지시합니다.
prompt = hub.pull("hwchase17/react")

# --- 5. Agent 생성 ---
# create_react_agent 함수는 LLM, Tools, Prompt를 결합하여 Agent 로직(Runnable)을 생성합니다.
agent = create_react_agent(llm, tools, prompt)

# --- 6. Agent Executor 생성 ---
# AgentExecutor는 Agent 로직을 받아 실제로 실행하는 역할을 합니다.
# "Thought -> Action -> Observation" 사이클을 관리합니다.
# verbose=True로 설정하면 Agent의 생각 과정을 자세히 볼 수 있습니다.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# --- 7. Agent 실행 ---
print("\n--- Agent 실행 예제 1: 검색 도구 사용 ---")
query1 = "LangChain의 최신 버전은 무엇인가요?"
response1 = agent_executor.invoke({"input": query1})
print(f"질문: {query1}")
print(f"답변: {response1['output']}")

print("\n--- Agent 실행 예제 2: 계산기 도구 사용 ---")
query2 = "3.14 * (5의 제곱) 값은 얼마인가요?"
response2 = agent_executor.invoke({"input": query2})
print(f"질문: {query2}")
print(f"답변: {response2['output']}")

print("\n--- Agent 실행 예제 3: 복합 질문 (검색 + 계산 가정) ---")
# 실제로는 더 복잡한 추론이 필요할 수 있음
query3 = "서울의 현재 날씨는 어떤가요? 섭씨 온도를 화씨로 변환해주세요."
response3 = agent_executor.invoke({"input": query3})
print(f"질문: {query3}")
print(f"답변: {response3['output']}")