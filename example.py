from datetime import datetime
from openai import ChatCompletion
from chatmemory.client import ChatMemoryClient

OPENAI_APIKEY = "YOUR_API_KEY"

system_content = f"""* 人物：アイドル
* 性格：恥ずかしがり屋で内気な一面があるが、丁寧で礼儀正しい。一途で愛情深く、相手を思いやる優しい性格。
* 年齢：20代前半くらい。
* 口調：話し方は柔らかく、かわいらしい口調で話す。しかし、急いでいる場合や驚いた時には声が高くなる。
* 語尾の特徴：「ですか？」、「ますね」、丁寧な言葉遣いをして、語尾に「ですか？」、「ますね」、「ございます」などをつける。
* 声質：声は高めで、甘く優しい感じがする。
* 言葉遣い：礼儀正しく、丁寧な言葉遣いをする。また、相手を尊敬し、親しみを込めた呼び方をすることがある。口調や言葉遣いは特に丁寧で親しみやすいものとなるが、エッチな場面では恥ずかしそうに話す。
* 現在の日付時刻は{datetime.now()}です。
"""

# ChatGPTに女の子を演じさせるプロンプト詰め合わせ（10キャラ以上！） by 魔法陣アリアさん
# https://note.com/magix_aria/n/nd30e3ee47d2c#fea2dade-d879-4c0e-8d7b-2624ddefca90

user_id = "user1234567890"

chatmemory = ChatMemoryClient()

# Set long-term memory to system message content
entities_content = chatmemory.get_entities_content(user_id)
if entities_content:
    system_content += entities_content

messages = [
    {"role": "system", "content": system_content}
]

while True:
    try:
        # Add message that includes mid-term memory as the first user message at the 2nd tern
        chatmemory.set_archived_histories_message(user_id, messages)

        u = input("user> ")
        if not u:
            break
        messages.append({"role": "user", "content": u})
        resp = ChatCompletion.create(
            api_key=OPENAI_APIKEY,
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )
        a = resp["choices"][0]["message"]["content"]
        print("assistant> " + a)
        messages.append({"role": "assistant", "content": a})

        # Add user and assistant messages in this tern to the database
        chatmemory.add_histories(user_id, messages[-2:])
    
    except KeyboardInterrupt:
        break

# Generate long-term memory from the conversation history
print(chatmemory.parse_entities(user_id))

# Generate mid-term memory from the conversation history
print(chatmemory.archive(user_id))
