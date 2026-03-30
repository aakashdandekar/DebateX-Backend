import traceback
from langchain_core.tracers import context
import traceback
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from bson import ObjectId
from src.database.db import database
from src.models.schemas import User, Login
from src.core.auth import hash, check_hash, get_current_user, create_access_token
from src.services.ai import modelResponse, judge_debate, find_topic, modelConversationSimulation_for, modelConversationSimulation_against, judge_debate_model

app = FastAPI(
    servers=[{"url": "http://localhost:8000"}]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static", html=True, check_dir=False), name="static")

@app.get("/")
async def serve_frontend(request: Request):
    return RedirectResponse(url="http://localhost:5173")

@app.post('/register')
async def register_user(user: User, request: Request):
    try:
        collection = database["user"]

        exist = await collection.find_one({
            "$or": [
                {"email": user.email},
                {"name": user.name}
            ]
        })

        if exist:
            raise HTTPException(status_code=400, detail="User already exists!")

        password = hash(password=user.password)

        result = await collection.insert_one({
            "name": user.name,
            "email": user.email,
            "password": password,
            "points": 0,
            "created_at": datetime.now(tz=timezone.utc)
        })

        token = create_access_token(str(result.inserted_id))

        return {"message": "User registered successfully"}

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post('/login')
async def login_user(login: Login):
    try:
        collection = database["user"]
        user = await collection.find_one({
            "email": login.email
        })

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if check_hash(login.password, user["password"]):
            token = create_access_token(str(user["_id"]))
        else:
            raise HTTPException(status_code=401, detail="Invalid Credentials")

        return {"access_token": token}

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get('/get-topic')
async def get_topic():
    return await find_topic()

#User VS AI
@app.patch('/system/start-debate')
async def start_system_debate(topic: str, role: str, current_user: str = Depends(get_current_user)):
    try:
        context_history_collection = database["context_history"]

        exist = await context_history_collection.find_one({"user_id": current_user})
        if exist:
            raise HTTPException(status_code=400, detail="Debate is already active")

        await context_history_collection.update_one(
            {"user_id": current_user},
            {
                "$set": {
                    "topic": topic,
                    "role": role,
                    "active_debate": topic,
                    "context": "",
                    "created_at": datetime.now(tz=timezone.utc)
                }
            },
            upsert=True
        )

        return {"message": "Debate started"}

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.patch('/system/debate/system-response')
async def system_response(user_response: str, current_user: str = Depends(get_current_user)):
    try:
        response = await modelResponse(argument=user_response, user_id=current_user)

        context_history_collection = database["context_history"]
        user = await context_history_collection.find_one({"user_id": current_user})
        
        if user:
            current_context = user.get("context", "")
            updated_context = f"{current_context}\nUser: {user_response}\nSystem: {response}"
            
            await context_history_collection.update_one(
                {"user_id": current_user},
                {"$set": {"context": updated_context}}
            )

        return {
            "System-response": response
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.delete("/system/end-debate")
async def end_system_debate(current_user: str = Depends(get_current_user)):
    try:
        user_collection = database["user"]
        context_history_collection = database["context_history"]

        verdict = await judge_debate(current_user)

        if verdict.get("winner", "") == "user":
            await user_collection.update_one(
                {"_id": ObjectId(current_user)},
                {"$inc": {"points": 2}},
                upsert=True
            )
        else:
            await user_collection.update_one(
                {"_id": ObjectId(current_user)},
                {"$inc": {"points": -1}},
                upsert=True
            )

        await context_history_collection.delete_one({"user_id": current_user})

        return verdict

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

#AI VS AI
@app.post('/system/spectator-mode/start')
async def spectator_mode(
    topic: str,
    current_user: str = Depends(get_current_user)
):
    try:
        collection = database['spectator-mode']
        result = await collection.insert_one({
            "user_id": current_user,
            "topic": topic,
            "context": ""
        })

        return {
            "message": f"Spectator Mode initiated, {result.inserted_id}"
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.patch('/system/spectator-mode/debate')
async def debate_ai(limit:int = 20, current_user: str = Depends(get_current_user)):
    try:
        collection = database['spectator-mode']
        chat = await collection.find_one({"user_id": current_user})

        topic = chat.get("topic", "")
        context = chat.get("context", "")
        
        if limit % 2 == 0:
            response = await modelConversationSimulation_for(
                context=context,
                topic=topic,
                argument="Based on context and topic argue in favor for the topic"
            )
            context += f"\nModel1: {response}"
        
        else:
            response = await modelConversationSimulation_against(
                context=context,
                topic=topic,
                argument="Based on context and topic argue in favor against the topic"
            )
            context += f"\nModel2: {response}"
        
        await collection.update_one(
            {"user_id": current_user},
            {"$set": {"context": context}}
        )

        if limit == 0:
            return {"response": response}

        return {"response": response, "limit_left": limit - 1}

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.delete('/system/spectator-mode/end')
async def end_ai_debate(current_user: str = Depends(get_current_user)):
    try:
        judgement = await judge_debate_model(user_id=current_user)
        
        collection = database['spectator-mode']
        await collection.delete_one({"user_id": current_user})

        return {"response": judgement}

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

#Leaderboard
@app.get('/api/leader-board')
async def get_leaderboard(current_user: User = Depends(get_current_user)):
    try:
        collection = database['user']

        user = await collection.find_one({"_id": ObjectId(current_user)})
        if not user:
            raise HTTPException(status_code=400, detail="User not found")

        points = user.get("points", 0)

        ranking = await collection.find(
            {"points": {"$gt": 10}}
        ).sort("points", -1).limit(50).to_list(length=None)

        for user in ranking:
            user["_id"] = str(user["_id"])

        return {
            "user_points": points,
            "leader_board": ranking
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")