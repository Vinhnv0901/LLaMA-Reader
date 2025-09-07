import streamlit as st

from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter

# import pinecone

from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from huggingface_hub import notebook_login
import textwrap
import sys
import os
import torch
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()
token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')



st.title("LLama Reader")

urls = ['https://www.mosaicml.com/blog/mpt-7b',
    'https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models',
    'https://lmsys.org/blog/2023-03-30-vicuna/']
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()






text_splitter=CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
text_chunks=text_splitter.split_documents(data)
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.from_documents(text_chunks, embeddings)

model = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="auto",
    quantization_config=bnb_config,
    token=token
)

pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})




qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())


query = st.chat_input("Ask me anything: ") 
if query:
    response = qa.run(query)
    st.write(response)








# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
# docs = text_splitter.split_documents(data)
# all_splits = docs
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# llm = OpenAI(temperature=0.4, max_tokens=500)










# prompt = query

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )


# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )


# if query:
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#     response = rag_chain.invoke({"input": query})
#     print(response["answer"])

#     st.write(response["answer"])