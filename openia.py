from openai import OpenAI

# Your API key
api_key = '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'

client = OpenAI(api_key=api_key)

def ask_question(question):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model="gpt-3.5-turbo",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print("An error occurred:", e)
        return None  # Or handle the error as needed


def main():
    print("Welcome to the Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Bye! Have a great day.")
            break

        response = ask_question(user_input)
        if response:
            print("Chatbot:", response)


if __name__ == "__main__":
    main()
