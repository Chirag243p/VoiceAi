from fastapi import FastAPI, HTTPException
import speech_recognition as sr
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

@app.post("/speech-to-text")
async def speech_to_text():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": text}]
        )

        chatgpt_answer = response['choices'][0]['message']['content'].strip()

        return {"text": text, "chatgpt_response": chatgpt_answer}

    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Could not understand the audio.")
    except sr.RequestError:
        raise HTTPException(status_code=500, detail="Speech recognition service error.")
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
