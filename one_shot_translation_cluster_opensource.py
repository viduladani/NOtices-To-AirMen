# Adapted from OpenAI's Vision example 
from openai import OpenAI
import pandas as pd
import json
import fitz  # PyMuPDF for PDF extraction

# with open('one_shot_prompts.json', 'r') as file:
#     prompts = json.load(file)


def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Load rulebooks based on keyword detection
rulebook_paths = {
    "ficon": "ficon.pdf",      # Replace with your actual PDF file path for Rulebook 1
    "apron": "apron.pdf",      # Replace with your actual PDF file path for Rulebook 2
    "obst": "obst.pdf"  # Replace with your actual PDF file path for Rulebook 3
}
#Replace with your actual
with open('updated_clusterprompts.json', 'r') as file:
    prompts = json.load(file)

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

df = pd.read_csv('prepocessed_NOTAM_pairs.csv')

rulebook_content = {
    keyword: extract_text_from_pdf(path)
    for keyword, path in rulebook_paths.items()
}

df_out = pd.DataFrame(columns=df.columns)
responses = []
for input in range(0, len(df["NOTAM "])):
  
    messages = [
      {
        "role": "system",
        "content": "You are a translator. Your task is to translate NOTAM into English. You will give only one translation. Don't give key value pair, return only a normal English statement. Don't give part-wise translation, return complete translation as one single statement.",
      }
    ]


    notam_message = df["Final_NOTAM"][input].lower()
    # rulebook_included = False
    for keyword, content in rulebook_content.items():
        if keyword in notam_message:
            messages.append({
                "role": "system",
                "content": f"Give translation of all words in message. Rules for translation: {content}."
            })
            #rulebook_included = True
            #break



    for prompt in prompts[df["Notam Type "][input]][str(int(df["Label"][input]))]:
      messages.append(prompt)
    messages.append({
      "role": "user",
      "content": df["Final_NOTAM"][input] + ".",
      # "content": "Translate to English: " + input + ".",
    })

    completion = client.chat.completions.create(
      #model="lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF",
      #model="lmstudio-ai/gemma-2b-it-GGUF",
      #model="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
      model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
      messages=messages,
      temperature=0.1,
      max_tokens=4096
      #stream=True
    )
    response = ""
    print("\n==================" + str(input) + "==================\n")
    print(df["Notam Type "][input] + "\n")
    response=completion.choices[0].message.content
    print(response)
    # for chunk in completion:
    #   if chunk.choices[0].delta.content:
    #     response += chunk.choices[0].delta.content + " "
    #     print(chunk.choices[0].delta.content, end="", flush=True)
    temp_df = df.loc[[input]]  # This keeps it as a DataFrame
    df_out = pd.concat([df_out, temp_df], ignore_index=True)
    
    #df_out = df_out.append(df.loc[input], ignore_index=True)
    responses.append(response)

df_out["machine_translation_zero_shots"] = responses
df_out.to_csv("\\transalation_cluster_Mistral.csv")
print("end")
    