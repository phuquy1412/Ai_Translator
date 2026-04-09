import requests

def translate(text, model="translategemma:4b"):
    prompt = f"Dịch đoạn text sau sang tiếng Việt, chỉ trả về bản dịch, không giải thích:\n{text}"
    response = requests.post("http://localhost:11434/api/generate",
                             json={
                                 "model":model, "prompt":prompt,"stream":False
                                 }
                        )
    result = response.json()
    #print(result)
    return result["response"]
text = "武術をエンジョイしてるのが気に入らねーか?"
print(f"Text gốc: {text}")
print(f"Bản dịch: {translate(text)}")