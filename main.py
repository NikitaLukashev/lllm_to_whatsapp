from dotenv import load_dotenv
from twilio.rest import Client
from fastapi import FastAPI
from fastapi import Request
import numpy as np
import faiss

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os

from context import chunks, context

load_dotenv()
print(len(context))
for i in chunks:
    print(len(i))
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
api_key = os.environ["MISTRAL_API_KEY"]
host_phone = os.environ["HOST_PHONE"]
guest_phone = os.environ["GUEST_PHONE"]


twilio_client = Client(account_sid, auth_token)
mistral_client = MistralClient(api_key=api_key)


app = FastAPI()

with_context = True





@app.get("/")
async def index():
    return {"message": "hello"}


def send_message(body_text):
    twilio_client.messages.create(
        from_=f"whatsapp:{guest_phone}", body=body_text, to=f"whatsapp:{host_phone}"
    )



@app.post("/message")
async def reply(request: Request):
    message = await request.body()
    res = dict(s.split(b"=") for s in message.split(b"&"))[b'Body'].decode("utf-8")
    if with_context:
        index1 = add_context(chunks)[0]
        resp = predict(res, index1, chunks)

    else:
        resp = predict(res, "index", chunks)

    return send_message(resp)

def get_text_embedding(input):
    embeddings_batch_response = mistral_client.embeddings(
        model="mistral-embed",
        input=input
    )
    return embeddings_batch_response.data[0].embedding



def add_context(chunks):

    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    d = text_embeddings.shape[1]
    index1 = faiss.IndexFlatL2(d)
    index1.add(text_embeddings)
    added_contextss = "yes"
    return index1, added_contextss

def run_mistral(user_message, model="mistral-large-latest"):
    messages = [ ChatMessage(role="user", content=user_message) ]
    chat_response = mistral_client.chat(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content


def predict(msg, myindex, chunks):


    if with_context:
        tmp1 = question_embedding(msg)
        D, I = myindex.search(tmp1, k=5)
        retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
        prompt_rag = f"""
            Context information is below.
            ---------------------
            {retrieved_chunk}
            ---------------------
            Given the context information and not prior knowledge, 
            answer the query in question's language and in 
            less than 300 characters.
            Query: {msg}
            Answer:
            """

        tmp2 = run_mistral(prompt_rag)
    else:

        prompt_empty = f"""
           Answer the query in question's language and in 
           less than 300 characters..
           Query: {msg}
           Answer:
           """

        tmp2 = run_mistral(prompt_empty)

    return tmp2





def question_embedding(msg):
    question_embeddings = np.array([get_text_embedding(msg)])
    return question_embeddings
