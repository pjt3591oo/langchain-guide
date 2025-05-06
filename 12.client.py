import requests

# invoke: 단일 입력에 대한 결과 반환
response_invoke = requests.post(
    "http://localhost:8000/rag-chain/invoke",
    json={'input': 'LangServe는 무엇인가요?'} # Runnable의 입력 구조에 맞춰 json 전달
)
print("--- Invoke 결과 ---")
# LangServe는 출력 타입을 자동으로 파악하여 json으로 반환합니다.
# StrOutputParser를 사용했으므로 'output' 키에 문자열 결과가 들어있습니다.
print(response_invoke.json())


# batch: 여러 입력에 대한 결과 리스트 반환
response_batch = requests.post(
    "http://localhost:8000/rag-chain/batch",
    json={'inputs': ['LangServe는 무엇인가요?', 'LangSmith는 무엇인가요?']}
)
print("\n--- Batch 결과 ---")
print(response_batch.json())


# stream: 결과를 스트리밍 형태로 받기 (청크 단위)
response_stream = requests.post(
    "http://localhost:8000/rag-chain/stream",
    json={'input': 'LangServe의 장점은 무엇인가요?'},
    stream=True # 스트리밍 요청 설정
)
print("\n--- Stream 결과 ---")
for chunk in response_stream.iter_content(chunk_size=None):
    if chunk:
        # 스트리밍 응답은 일반적으로 byte 형태이므로 디코딩 필요
        # 출력 형식에 따라 파싱 방법이 다를 수 있음 (기본은 문자열 청크)
        try:
            # LangServe의 스트리밍은 Server-Sent Events (SSE) 형식을 따를 수 있음
            # 간단한 텍스트 스트림으로 가정하고 출력
            print(chunk.decode('utf-8'), end="")
        except UnicodeDecodeError:
            print("[binary chunk]", end="") # 디코딩 불가 청크
print("\n-------------------")