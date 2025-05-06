from langchain.prompts import ChatPromptTemplate

# 채팅 메시지 템플릿 정의
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {ai_name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"), # 사용자의 실제 입력이 들어갈 부분
    ]
)

# 템플릿에 변수 값 채워서 메시지 리스트 생성
messages = chat_template.format_messages(ai_name="Bob", user_input="What is your name?")

print(messages)
# 출력 예시:
# [SystemMessage(content='You are a helpful AI bot. Your name is Bob.'),
#  HumanMessage(content='Hello, how are you doing?'),
#  AIMessage(content="I'm doing well, thanks!"),
#  HumanMessage(content='What is your name?')]

# 위 messages를 chat_model.invoke()에 전달하면 됨
ai_response = chat_model.invoke(messages)
print(ai_response.content)
# 출력 예시: My name is Bob.