from flask import Flask, request, jsonify
import speech_recognition as sr

app = Flask(__name__)

@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return jsonify({"text": text})
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand the audio."})
    except sr.RequestError:
        return jsonify({"error": "Speech recognition service error."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
