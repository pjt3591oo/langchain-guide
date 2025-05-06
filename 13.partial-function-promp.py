from datetime import datetime
from langchain_core.prompts import PromptTemplate

# 현재 시간대에 따라 인사말을 반환하는 함수 정의
def get_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "좋은 아침"
    elif 12 <= hour < 18:
        return "좋은 오후"
    elif 18 <= hour < 22:
        return "좋은 저녁"
    else:
        return "안녕히 주무세요"

# 함수를 사용한 부분 변수가 있는 프롬프트 템플릿 정의
prompt = PromptTemplate(
    template="{greeting}하세요! 오늘 추천드리는 활동은 {activity}입니다.",
    input_variables=["activity"],  # 사용자 입력이 필요한 변수
    partial_variables={"greeting": get_greeting}  # 함수를 통해 동적으로 값을 생성하는 부분 변수
)

# 'activity' 변수만 입력하여 시간대에 맞는 인사말이 포함된 문장 생성
print(prompt.format(activity="산책하기"))