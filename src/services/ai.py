import re
import os
import json
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from Backend.database.db import database
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

groq_llama3_instant_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

groq_llama3_versatile_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

groq_openai_gpt_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#Finding Topic
async def find_topic():
    prompt = PromptTemplate(
        template="""
            Generate one debate topic and describe it in exactly one line.
            Output plain text only.
            Do not include any formatting, explanations, or reasoning.
            The output should contain only the topic followed by a brief one-line description.
        """
    )

    chain = prompt | groq_llama3_instant_llm | StrOutputParser()

    return await chain.ainvoke({})

#Spectator Mode: AI vs AI
async def modelConversationSimulation_for(topic: str, context: str, argument: str) -> str:
    docs = splitter.create_documents([context]) if context else []
    if docs:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type=docs,
            search_kwargs={'k': 5}
        )

        retrieved_docs = retriever.invoke(response)
        context_summary = '\n\n'.join([d.page_content for d in retrieved_docs])

        vectorstore.delete_collection()
    else:
        context_summary = ""

        template = """
        Topic: {topic}
        Your Role: For
        You are a professional debater. Your job is to argue AGAINST whatever the user says.
        No matter what they claim, challenge it with strong counter-arguments.
        Be direct, logical, and confident. Keep it to 3-4 sentences.

        Context for debate is {context_summary}
        
        User says: {argument}
        
        Counter-argument:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['topic', 'argument', 'context_summary']
    )

    chain = prompt | groq_llama3_versatile_llm | StrOutputParser()
    response = await chain.ainvoke({"topic": topic, "argument": argument, "context_summary": context_summary})

    context += f"\nModel 1: {response}"

    return response

async def modelConversationSimulation_against(topic: str, context: str, argument: str) -> str:
    docs = splitter.create_documents([context]) if context else []
    if docs:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type=docs,
            search_kwargs={'k': 5}
        )

        retrieved_docs = retriever.invoke(response)
        context_summary = '\n\n'.join([d.page_content for d in retrieved_docs])

        vectorstore.delete_collection()
    else:
        context_summary = ""

        template = """
        Topic: {topic}
        Your Role: Against
        You are a professional debater. Your job is to argue AGAINST whatever the user says.
        No matter what they claim, challenge it with strong counter-arguments.
        Be direct, logical, and confident. Keep it to 3-4 sentences.

        Context for debate is {context_summary}
        
        User says: {argument}
        
        Counter-argument:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['topic', 'argument', 'context_summary']
    )

    chain = prompt | groq_llama3_versatile_llm | StrOutputParser()
    response = await chain.ainvoke({"topic": topic, "argument": argument, "context_summary": context_summary})

    context += f"\nModel 2: {response}"

    return response


#Competative Mode: User vs AI
async def modelResponse(argument: str, user_id: str) -> str:
    context_collection = database["context_history"]

    result = await context_collection.find_one({"user_id": user_id})
    context = result.get("context", "") if result else ""

    topic = result.get("topic", "")
    role = result.get("role", "")

    role = "for" if role == "against" else "against"

    docs = splitter.create_documents([context]) if context else []
    if docs:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        retrieved_docs = retriever.invoke(argument)
            
        context_summary = "\n\n".join([d.page_content for d in retrieved_docs])

        vectorstore.delete_collection()
    else:
        context_summary = ""

    template = """
        Topic: {topic}
        Your Role: {role}
        You are a professional debater. Your job is to argue AGAINST whatever the user says.
        No matter what they claim, challenge it with strong counter-arguments.
        Be direct, logical, and confident. Keep it to 3-4 sentences.

        Context for debate is {context_summary}
        
        User says: {argument}
        
        Counter-argument:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['role', 'topic', 'argument', 'context_summary']
    )

    chain = prompt | groq_llama3_versatile_llm | StrOutputParser()
    response = await chain.ainvoke({"role": role, "topic": topic, "argument": argument, "context_summary": context_summary})

    return response

#Judgement System
async def judge_debate(user_id: str):
    context_collection = database["context_history"]
    result = await context_collection.find_one({"user_id": user_id})

    if not result:
        raise ValueError(f"No debate session found for user_id: {user_id}")

    topic = result.get("active_debate", "")
    context = result.get("context", "")

    template = """
        You are a strict debate judge with years of experience.
        You will be given a debate conversation between a User and an AI debater.
        
        Evaluate both sides based on:
        1. Logic & Reasoning - How well-structured are the arguments?
        2. Evidence & Facts - Are claims backed up?
        3. Persuasiveness - How convincing is each side?
        4. Clarity - How clear and concise are the points?

        Topic of Debate:
        {topic}

        Context:
        {context}

        Respond ONLY in this exact JSON format:
        {{
            "winner": "user" or "system",
            "user_score": <score out of 10>,
            "user_feedback": "<what the user did well and poorly>",
            "reasoning": "<2-3 sentences explaining why the winner won>"
        }}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['topic', 'context']
    )

    vectorstore = None

    docs = splitter.create_documents([context]) if context else []
    if docs:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        retrieved_docs = retriever.invoke(topic)

        context = "\n\n".join([d.page_content for d in retrieved_docs])
    else:
        context = ""

    chain = prompt | groq_openai_gpt_llm | StrOutputParser()

    response = await chain.ainvoke({"topic": topic, "context": context})
    response = re.sub(r"```json|```", "", response).strip()

    if vectorstore is not None:
        vectorstore.delete_collection()

    return json.loads(response)
