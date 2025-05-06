from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# .env 파일에서 환경 변수 로드 (API 키 관리)
# .env 파일에 OPENAI_API_KEY="your_api_key" 형식으로 저장
load_dotenv()

# ChatOpenAI 모델 초기화
# temperature는 생성 결과의 무작위성 조절 (0: 결정적, 높을수록 무작위)
chat_model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0.7, 
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)

# 모델에 메시지 전달 및 응답 받기
# SystemMessage: 챗봇의 역할이나 지침 설정
# HumanMessage: 사용자의 입력
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to Korean."),
    HumanMessage(content="Translate this sentence from English to Korean: I love programming."),
]

# invoke 메서드로 단일 응답 얻기
ai_response = chat_model.invoke(messages)

print(ai_response)
# 출력 예시: AIMessage(content='저는 프로그래밍을 사랑합니다.', response_metadata=...)
print(ai_response.content)
# 출력 예시: 저는 프로그래밍을 사랑합니다.