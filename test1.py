# assistant.py
import time
from collections import deque

import numpy as np
import sounddevice as sd
import torch
import ollama
import webbrowser
import subprocess
from faster_whisper import WhisperModel
from ddgs import DDGS

# ---------- Настройки ----------
SAMPLE_RATE   = 16000
LLM_MODEL     = "llama3.1"
CONTEXT_TOKENS = 32768          # память диалога: 32к токенов
TTS_SPEAKER   = "aidar"         # один мужской голос
TTS_RATE      = 48000

# команды, которые выполняются БЕЗ подтверждения
SAFE_COMMANDS = ("dir", "ls", "echo", "date", "whoami", "pwd", "ver", "uname", "hostname", "ipconfig")

# ---------- 1. Распознавание речи ----------
stt = WhisperModel("base", device="cpu", compute_type="int8")

def listen_until_silence(max_duration=20, silence_duration=1.2,
                         threshold=0.02, wait_timeout=10) -> str:
    """Слушает микрофон: ждёт начало речи, пишет до наступления тишины."""
    print("🎤 Говорите...")
    chunk_size = int(SAMPLE_RATE * 0.1)          # 100 мс
    prefetch = deque(maxlen=5)                    # предбуфер ~0.5 сек
    frames = []
    started = False
    silence_start = None
    speech_start = None
    wait_start = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            volume = float(np.abs(chunk).mean())
            now = time.time()

            if not started:
                prefetch.append(chunk.copy())
                if volume > threshold:
                    started = True
                    speech_start = now
                    frames = list(prefetch)       # не потерять начало фразы
                    frames.append(chunk.copy())
                elif now - wait_start > wait_timeout:
                    return ""                      # ничего не сказали
                continue

            frames.append(chunk.copy())
            if volume > threshold:
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = now
                elif now - silence_start > silence_duration:
                    break                           # тишина наступила
            if now - speech_start > max_duration:
                break                               # защита от слишком длинной речи

    if not frames:
        return ""
    audio = np.concatenate(frames).flatten()
    segments, _ = stt.transcribe(audio, language="ru")
    text = " ".join(s.text for s in segments).strip()
    print(f"Вы: {text}")
    return text

# ---------- 2. Синтез речи (один мужской голос) ----------
tts, _ = torch.hub.load('snakers4/silero-models', 'silero_tts')

def speak(text: str):
    print(f"🤖 {text}")
    audio = tts.apply_tts(text=text, speaker=TTS_SPEAKER, sample_rate=TTS_RATE)
    sd.play(audio, TTS_RATE)
    sd.wait()

# ---------- 3. Инструменты ----------
def web_search(query: str) -> str:
    results = list(DDGS().text(query, region="ru-ru", max_results=4))
    if not results:
        return "Ничего не найдено."
    return "\n".join(f"{r['title']}: {r['body']}" for r in results)

def open_site(url: str) -> str:
    webbrowser.open(url if url.startswith("http") else "https://" + url)
    return f"Открываю {url}"

def _confirm(question: str) -> bool:
    speak(question)
    answer = listen_until_silence(max_duration=8, silence_duration=1.0)
    return any(w in answer.lower() for w in ("да", "ага", "конечно", "ок", "давай", "выполни"))

def run_command(cmd: str) -> str:
    cmd_stripped = cmd.strip().lower()
    is_safe = any(cmd_stripped.startswith(s) for s in SAFE_COMMANDS)
    if not is_safe and not _confirm(f"Выполнить команду {cmd}? Скажите да или нет."):
        return "Команда отклонена пользователем."
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        return out.stdout or out.stderr or "Выполнено"
    except Exception as e:
        return f"Ошибка: {e}"

functions = {"web_search": web_search, "open_site": open_site, "run_command": run_command}

tools = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Поиск в интернете для свежих фактов и новостей.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "open_site",
        "description": "Открыть сайт в браузере.",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Выполнить системную команду в терминале.",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}},
]

# ---------- 4. Мозг + память диалога ----------
SYSTEM = ("Ты — голосовой ассистент. Отвечай кратко (1-3 предложения), по-русски, "
          "без разметки. Если нужен интернет или действие — вызывай инструмент.")

history = [{"role": "system", "content": SYSTEM}]
MAX_CHARS = CONTEXT_TOKENS * 3   # грубая защита: ~3 символа на токен

def _trim_history():
    system = history[0]
    rest = history[1:]
    total = sum(len(str(m.get("content", ""))) for m in rest)
    while total > MAX_CHARS and len(rest) > 2:
        removed = rest.pop(0)
        total -= len(str(removed.get("content", "")))
    history[:] = [system] + rest

def think(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    _trim_history()

    resp = ollama.chat(model=LLM_MODEL, messages=history, tools=tools,
                       options={"num_ctx": CONTEXT_TOKENS})
    msg = resp["message"]

    if msg.get("tool_calls"):
        results = []
        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            print(f"⚙️ Вызов: {name}({args})")
            results.append(str(functions[name](**args)))
        history.append(msg)
        history.append({"role": "tool", "content": "\n".join(results)})
        resp = ollama.chat(model=LLM_MODEL, messages=history,
                           options={"num_ctx": CONTEXT_TOKENS})
        msg = resp["message"]

    history.append({"role": "assistant", "content": msg["content"]})
    return msg["content"]

# ---------- 5. Главный цикл ----------
if __name__ == "__main__":
    speak("Привет! Я слушаю.")
    while True:
        text = listen_until_silence()
        if not text:
            continue
        if any(w in text.lower() for w in ("стоп", "выход", "пока")):
            speak("До свидания!")
            break
        try:
            speak(think(text))
        except Exception as e:
            speak(f"Ошибка: {e}")