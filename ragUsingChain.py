from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled,NoTranscriptFound
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
st.title("Youtube Chatbot")
video_id=st.text_input("enter your Video ID")

query=st.text_input("enter the question")

if(st.button("Answer")):

    try:
        ytt_api=YouTubeTranscriptApi()
        transcript_list=ytt_api.fetch(video_id,languages=['en'])
        transcript=" ".join(snippet.text for snippet in transcript_list)
    except TranscriptsDisabled:
        print("transcript is not available")
    except NoTranscriptFound:
        print("no transcript is found in the request language(s)")

    spillter=RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )

    chunks=spillter.create_documents([transcript])

    vector_store=FAISS.from_documents(chunks,HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))

    retriever=vector_store.as_retriever(search_kwargs={'k':6})



    def format_docs(retrieve_docs):
        context="\n\n".join(doc.page_content for doc in retrieve_docs )
        return context

    augment_chain=RunnableParallel({
        "context":retriever|RunnableLambda(format_docs),
        "question":RunnablePassthrough()
    })


    template=PromptTemplate(
        template="""
    you are a helpful assistant.
    answer only from the provided transcript context.
    if context is insufficient then simply answer the context is insufficient for the aksed question
    context:{context}
    question:{question}"""
    )
    # llm=HuggingFaceEndpoint(
    #     repo_id="deepseek-ai/DeepSeek-V4-Pro",
    #     provider="deepinfra",
    #     task="text-generation",
    #     temperature=0.0
    # )
    model=ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    parser=StrOutputParser()
    generation_chain=template|model|parser

    final_chain=augment_chain|generation_chain

    result=final_chain.invoke(query)
    st.write(result)