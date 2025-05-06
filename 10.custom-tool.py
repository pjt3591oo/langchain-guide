import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, tool # tool 데코레이터 임포트
from langchain import hub
from dotenv import load_dotenv

load_dotenv()
# --- 1. 환경 변수 설정 (필요시) ---
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# --- 2. LLM 준비 ---
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model_name="l3-8b-stheno-v3.1-iq-imatrix",
    temperature=0.7
)

# --- 3. 커스텀 도구 정의 ---
@tool
def get_current_datetime(query: str) -> str:
  """
  현재 날짜와 시간을 알아야 할 때 사용합니다.
  입력(query)은 사용하지 않지만, 형식상 필요합니다.
  현재 날짜와 시간을 문자열 형태로 반환합니다.
  예: '현재 시간은 2025년 5월 4일 오후 5시 55분입니다.'
  """
  now = datetime.now()
  # 사용자가 이해하기 쉬운 형태로 포맷팅 (한국 시간 기준 예시)
  return f"현재 시간은 {now.strftime('%Y년 %m월 %d일 %p %I시 %M분')}입니다."

# (선택) 예제: 간단한 사용자 프로필 조회 도구 (사내 시스템 연동 가정)
user_database = {
    "Alice": {"email": "alice@example.com", "department": "Engineering"},
    "Bob": {"email": "bob@sample.org", "department": "Marketing"}
}

@tool
def get_user_profile(username: str) -> str:
  """
  회사 직원의 이메일 주소나 부서 등 프로필 정보가 필요할 때 사용합니다.
  사용자 이름(username)을 입력받아 해당 사용자의 프로필 정보를 반환합니다.
  만약 사용자를 찾을 수 없으면, '사용자를 찾을 수 없습니다.' 라고 답합니다.
  """
  if username in user_database:
    profile = user_database[username]
    return f"{username}의 정보: 이메일={profile.get('email', 'N/A')}, 부서={profile.get('department', 'N/A')}"
  else:
    return f"사용자 '{username}'을(를) 찾을 수 없습니다."

# --- 4. 사용할 도구 리스트 정의 (커스텀 도구 포함) ---
tools = [get_current_datetime, get_user_profile] # @tool로 만든 함수는 바로 리스트에 추가 가능

# --- 5. Agent가 사용할 프롬프트 가져오기 ---
prompt = hub.pull("hwchase17/react")
print(prompt)
# --- 6. Agent 생성 ---
agent = create_react_agent(llm, tools, prompt)

# --- 7. Agent Executor 생성 ---
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 8. Agent 실행 (커스텀 도구 사용 유도) ---
print("\n--- Agent 실행 예제 1: 현재 시간 묻기 ---")
query1 = "지금 몇 시야?"
response1 = agent_executor.invoke({"input": query1})
print(f"질문: {query1}")
print(f"답변: {response1['output']}")

print("\n--- Agent 실행 예제 2: 직원 정보 묻기 ---")
query2 = "Alice의 이메일 주소 알려줘."
response2 = agent_executor.invoke({"input": query2})
print(f"질문: {query2}")
print(f"답변: {response2['output']}")

print("\n--- Agent 실행 예제 3: 없는 직원 정보 묻기 ---")
query3 = "Charlie의 부서는 어디야?"
response3 = agent_executor.invoke({"input": query3})
print(f"질문: {query3}")
print(f"답변: {response3['output']}")