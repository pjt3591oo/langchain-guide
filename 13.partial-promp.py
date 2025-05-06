from langchain_core.prompts import PromptTemplate

# 기본 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template("블록체인의 3대 요소는 {imte0}, {item1}, {item2} 입니다.")

# 'item0' 변수에 '블록' 값을 미리 지정하여 부분 포맷팅
partial_prompt0 = prompt.partial(imte0="블록")
# 'item1' 변수에 '트랜잭션' 값을 미리 지정하여 부분 포맷팅
partial_prompt1 = partial_prompt0.partial(item1="트랜잭션")

# 나머지 'item2' 변수만 입력하여 완전한 문장 생성
print(partial_prompt1.format(item2="상태"))

# 프롬프트 초기화 시 부분 변수 지정
prompt = PromptTemplate(
    template="블록체인의 3대 요소는 {imte0}, {item1}, {item2} 입니다.",
    input_variables=["item2"],  # 사용자 입력이 필요한 변수
    partial_variables={"imte0": "블록", "item1": "트랜잭션"}  # 미리 지정된 부분 변수
)

# 남은 'element' 변수만 입력하여 문장 생성
print(prompt.format(item2="규소"))