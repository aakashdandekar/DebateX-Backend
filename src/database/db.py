import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL=os.getenv("DATABASE_URL")
DATABASE_NAME=os.getenv("DATABASE_NAME")

client = AsyncIOMotorClient(DATABASE_URL)
database = client[DATABASE_NAME]