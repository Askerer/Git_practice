from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = ""

def chat(messages):
    client = OpenAI()
    response = client.chat.completions.create(
        model = "gpt-4",
        messages = messages,
        max_tokens = 150
    )
    return response.choices[0].message.content

print("歡迎來到翻譯中心")

messages = [{"role":"system","content":"你是中翻英人員"}]

while True:
    user_input = input("   客戶 : ")
    if user_input.lower() == "bye":
        print("bye bye")
        break
    messages.append({"role":"user","content":user_input})
    response = chat(messages)
    print("翻譯中心 " + response)
    messages.append({"role":"assistant","content":response})