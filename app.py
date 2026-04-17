from flask import Flask, request, jsonify
import os
import tempfile
import subprocess
import requests

app = Flask(__name__)


def download_file(url: str, dest_path: str):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "ok": True,
        "service": "TalkingFlow Whisper Align",
        "message": "Servidor online."
    })


@app.route("/align", methods=["POST"])
def align():
    try:
        data = request.get_json(force=True) or {}

        text = (data.get("text") or "").strip()
        audio_url = (data.get("audio_url") or "").strip()
        language = (data.get("language") or "en").strip()

        if not text:
            return jsonify({"ok": False, "error": "Campo 'text' é obrigatório."}), 400

        if not audio_url:
            return jsonify({"ok": False, "error": "Campo 'audio_url' é obrigatório."}), 400

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "input_audio.mp3")
            output_dir = os.path.join(tmpdir, "out")
            os.makedirs(output_dir, exist_ok=True)

            download_file(audio_url, audio_path)

            cmd = [
                "whisperx",
                audio_path,
                "--model", "base",
                "--language", language,
                "--device", "cpu",
                "--compute_type", "int8",
                "--output_dir", output_dir,
                "--output_format", "json",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return jsonify({
                    "ok": False,
                    "error": "Falha ao executar whisperx.",
                    "stderr": result.stderr,
                    "stdout": result.stdout
                }), 500

            json_path = os.path.join(output_dir, "input_audio.json")
            if not os.path.exists(json_path):
                return jsonify({
                    "ok": False,
                    "error": "Arquivo JSON de saída não foi gerado."
                }), 500

            import json
            with open(json_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)

            words = []
            for segment in parsed.get("segments", []):
                for w in segment.get("words", []):
                    word = (w.get("word") or "").strip()
                    start = w.get("start")
                    end = w.get("end")

                    if word and start is not None and end is not None:
                        words.append({
                            "word": word,
                            "start": start,
                            "end": end
                        })

            return jsonify({
                "ok": True,
                "text": text,
                "language": language,
                "words": words
            })

    except requests.RequestException as e:
        return jsonify({"ok": False, "error": f"Erro ao baixar áudio: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)