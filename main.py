from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
from transformers import pipeline
from fastapi.responses import StreamingResponse

app = FastAPI()

# Разрешаем CORS с определенных доменов (Frontend URL)
origins = [
    "https://your-frontend-project.vercel.app",  # Заменить на свой Frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Разрешаем запросы с этих доменов
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы
    allow_headers=["*"],  # Разрешаем все заголовки
)

# Инициализация модели генерации изображения
generator = pipeline("text-to-image", model="CompVis/stable-diffusion-v-1-4-original")

class Prompt(BaseModel):
    prompt: str

@app.post("/api/generate")
async def generate_image(prompt: Prompt):
    # Генерация изображения
    image = generator(prompt.prompt)[0]

    # Преобразуем изображение в байты
    byte_io = BytesIO()
    image.save(byte_io, "PNG")
    byte_io.seek(0)

    return StreamingResponse(byte_io, media_type="image/png")
