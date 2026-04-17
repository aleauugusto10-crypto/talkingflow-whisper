from flask import Flask, request, jsonify
import whisper

app = Flask(__name__)

model = whisper.load_model("base")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files["audio"]
    file.save("audio.mp3")

    result = model.transcribe("audio.mp3")

    return jsonify({
        "text": result["text"]
    })

@app.route("/")
def home():
    return "TalkingFlow Whisper API rodando 🚀"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)