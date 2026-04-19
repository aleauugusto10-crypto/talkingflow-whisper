from flask import Flask, request, jsonify
import os
import json
import tempfile
import subprocess
import requests
import shutil
import sys

app = Flask(__name__)

LANGUAGE_MAP = {
    "English": "en",
    "Português": "pt",
    "Español": "es",
    "Français": "fr",
    "Deutsch": "de",
    "Italiano": "it",
    "Nederlands": "nl",
    "Svenska": "sv",
    "Norsk": "no",
    "Dansk": "da",
    "Suomi": "fi",
    "Polski": "pl",
    "Čeština": "cs",
    "Română": "ro",
    "Magyar": "hu",
    "Türkçe": "tr",
    "Ελληνικά": "el",
    "Русский": "ru",
    "Українська": "uk",
    "العربية": "ar",
    "עברית": "he",
    "हिन्दी": "hi",
    "বাংলা": "bn",
    "اردو": "ur",
    "தமிழ்": "ta",
    "తెలుగు": "te",
    "한국어": "ko",
    "日本語": "ja",
    "中文（简体）": "zh",
    "中文（繁體）": "zh",
    "ไทย": "th",
    "Tiếng Việt": "vi",
    "Bahasa Indonesia": "id",
    "Bahasa Melayu": "ms",
    "Filipino": "tl",
    "Swahili": "sw",
}


def log(*args):
    print(*args, flush=True)


def download_file(url: str, dest_path: str):
    with requests.get(url, stream=True, timeout=120) as r:
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


@app.route("/debug", methods=["GET"])
def debug():
    return jsonify({
        "ok": True,
        "python": sys.version,
        "ffmpeg_found": bool(shutil.which("ffmpeg")),
        "whisperx_found": bool(shutil.which("whisperx")),
        "port": os.environ.get("PORT"),
    })


@app.route("/align", methods=["POST"])
def align():
    try:
        log("=== /align START ===")

        data = request.get_json(force=True) or {}

        text = (data.get("text") or "").strip()
        audio_url = (data.get("audio_url") or "").strip()

        raw_language = (data.get("language") or "en").strip()
        language = LANGUAGE_MAP.get(raw_language, raw_language.lower())

        log("text:", text)
        log("audio_url:", audio_url)
        log("raw_language:", raw_language)
        log("mapped_language:", language)

        if not audio_url:
            return jsonify({
                "ok": False,
                "error": "Campo 'audio_url' é obrigatório."
            }), 400

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "input_audio.mp3")
            output_dir = os.path.join(tmpdir, "out")
            os.makedirs(output_dir, exist_ok=True)

            log("baixando áudio...")
            download_file(audio_url, audio_path)
            log("áudio baixado com sucesso:", audio_path, "size=", os.path.getsize(audio_path))

            cmd = [
                "whisperx",
                audio_path,
                "--model", "tiny",
                "--language", language,
                "--device", "cpu",
                "--compute_type", "int8",
                "--output_dir", output_dir,
                "--output_format", "json",
            ]

            log("executando whisperx:", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            log("whisperx finished")
            log("returncode:", result.returncode)
            log("stdout:", result.stdout[:4000] if result.stdout else "")
            log("stderr:", result.stderr[:4000] if result.stderr else "")

            if result.returncode != 0:
                return jsonify({
                    "ok": False,
                    "error": "Falha ao executar whisperx.",
                    "stderr": result.stderr,
                    "stdout": result.stdout
                }), 500

            json_path = os.path.join(output_dir, "input_audio.json")
            log("json_path esperado:", json_path)

            if not os.path.exists(json_path):
                return jsonify({
                    "ok": False,
                    "error": "Arquivo JSON de saída não foi gerado."
                }), 500

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

            log("words extraídas:", len(words))
            log("=== /align END OK ===")

            return jsonify({
                "ok": True,
                "text": text,
                "language": language,
                "words": words,
                "segments": parsed.get("segments", [])
            })

    except subprocess.TimeoutExpired:
        log("TimeoutExpired no whisperx")
        return jsonify({
            "ok": False,
            "error": "Tempo esgotado ao executar whisperx."
        }), 504

    except requests.RequestException as e:
        log("Erro ao baixar áudio:", str(e))
        return jsonify({
            "ok": False,
            "error": f"Erro ao baixar áudio: {str(e)}"
        }), 500

    except Exception as e:
        log("Erro inesperado:", str(e))
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)