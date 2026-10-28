# assistant.py — Финальная версия с расширенным словарём приложений
# pip install faster-whisper sounddevice ollama torch ddgs numpy requests beautifulsoup4 opencv-python pyautogui pillow pandas openpyxl pdf2image

import os
import re
import json
import time
import base64
import glob
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import sounddevice as sd
import torch
import ollama
import webbrowser
import subprocess
import requests
from bs4 import BeautifulSoup
from faster_whisper import WhisperModel
from ddgs import DDGS

# ================= НАСТРОЙКИ: 3 УРОВНЯ =================

ALICE_MODEL = "qwen3-vl:8b"            # ⚡ Алиса (мультимодальная)
ANALYST_MODEL = "qwen3.6:35b-a3b"      # 🧠 Аналитик (логика+файлы)
ARCHITECT_MODEL = "qwen3-coder:30b"    # 💎 Архитектор (кодинг)

ALICE_CONTEXT = 16384
ANALYST_CONTEXT = 16384
ARCHITECT_CONTEXT = 32768

SAMPLE_RATE = 16000
TTS_SPEAKER_RU = "aidar"
TTS_SPEAKER_EN = "en_0"
TTS_RATE = 48000

WORK_DIR = r"D:\prog\python_p\ii\domii"
DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(DESKTOP_DIR, exist_ok=True)

SAFE_COMMANDS = ("dir", "ls", "echo", "date", "whoami", "pwd",
                 "ver", "uname", "hostname", "ipconfig", "python --version",
                 "pip list", "pip show")

# Пути переменные
LOCALAPPDATA = os.environ.get('LOCALAPPDATA', '')
APPDATA = os.environ.get('APPDATA', '')
PROGRAM_FILES = r"C:\Program Files"
PROGRAM_FILES_X86 = r"C:\Program Files (x86)"
USER_HOME = os.path.expanduser("~")


# ================= 1. РАСПОЗНАВАНИЕ РЕЧИ =================
print("Загрузка STT...")
stt = WhisperModel("small", device="cpu", compute_type="int8")

def listen_until_silence(max_duration=30, silence_duration=1.2,
                         threshold=0.005, wait_timeout=15) -> str:
    print("🎤 Говорите...")
    chunk_size = int(SAMPLE_RATE * 0.1)
    prefetch = deque(maxlen=5)
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
                    frames = list(prefetch)
                    frames.append(chunk.copy())
                elif now - wait_start > wait_timeout:
                    return ""
                continue

            frames.append(chunk.copy())
            if volume > threshold:
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = now
                elif now - silence_start > silence_duration:
                    break
            if now - speech_start > max_duration:
                break

    if not frames:
        return ""
    audio = np.concatenate(frames).flatten()
    segments, _ = stt.transcribe(
        audio, language="ru", vad_filter=True,
        initial_prompt="Привет. Ассистент слушает команды на русском языке."
    )
    text = " ".join(s.text for s in segments).strip()
    print(f"Вы: {text}")
    return text


# ================= 2. СИНТЕЗ РЕЧИ =================
print("Загрузка TTS...")
try:
    tts_ru, _ = torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v4_ru')
except:
    tts_ru = None

try:
    tts_en, _ = torch.hub.load('snakers4/silero-models', 'silero_tts', language='en', speaker='v3_en')
except:
    tts_en = None

def detect_language(text: str) -> str:
    cyrillic = sum(1 for c in text if 'а' <= c.lower() <= 'я' or c.lower() == 'ё')
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return 'ru' if cyrillic >= latin else 'en'

def speak(text: str):
    print(f"🤖 {text}")
    lines = text.split('\n')
    clean_lines = []
    in_code_block = False
    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if not in_code_block and line.strip():
            clean_lines.append(line)
    
    clean_text = '\n'.join(clean_lines)
    clean_text = re.sub(r'[`*#]', '', clean_text)
    clean_text = re.sub(r'https?://\S+', '', clean_text)
    if not clean_text.strip():
        return
    
    lang = detect_language(clean_text)
    model = tts_ru if lang == 'ru' else tts_en
    speaker = TTS_SPEAKER_RU if lang == 'ru' else TTS_SPEAKER_EN
    if model is None:
        return
    
    chunks = re.split(r'(?<=[.!?])\s+', clean_text)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.endswith(('.', '!', '?')):
            chunk += "."
        if len(chunk) > 500:
            chunk = chunk[:500]
        try:
            audio = model.apply_tts(text=chunk, speaker=speaker, sample_rate=TTS_RATE)
            sd.play(audio, TTS_RATE)
            sd.wait()
        except:
            break


# ================= 3. ИНСТРУМЕНТЫ ДЛЯ АЛИСЫ =================

def analyze_image(image_path: str, question: str = "Опиши что видишь на картинке") -> str:
    if not os.path.exists(image_path):
        return f"Файл не найден: {image_path}"
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        resp = ollama.chat(model=ALICE_MODEL, messages=[{"role": "user", "content": question, "images": [image_data]}], options={"num_ctx": ALICE_CONTEXT})
        return resp.message.content if hasattr(resp, 'message') else str(resp)
    except Exception as e:
        return f"Ошибка анализа изображения: {e}"

def capture_and_analyze_camera(question: str = "Что ты видишь перед собой?") -> str:
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret: return "Не удалось получить кадр с камеры"
        temp_path = os.path.join(WORK_DIR, "_camera_capture.jpg")
        cv2.imwrite(temp_path, frame)
        result = analyze_image(temp_path, question)
        os.remove(temp_path)
        return f"[Камера]: {result}"
    except ImportError:
        return "Для работы с камерой установите opencv: pip install opencv-python"
    except Exception as e:
        return f"Ошибка камеры: {e}"

def take_screenshot_and_analyze(question: str = "Что ты видишь на экране?") -> str:
    try:
        import pyautogui
        screenshot = pyautogui.screenshot()
        temp_path = os.path.join(WORK_DIR, "_screen_capture.png")
        screenshot.save(temp_path)
        result = analyze_image(temp_path, question)
        os.remove(temp_path)
        return f"[Экран]: {result}"
    except ImportError:
        return "Для работы со скриншотами установите: pip install pyautogui pillow"
    except Exception as e:
        return f"Ошибка снятия скриншота: {e}"

def list_desktop_files() -> str:
    try:
        desktop_path = DESKTOP_DIR
        if not os.path.exists(desktop_path): return f"Папка рабочего стола не найдена: {desktop_path}"
        items = []
        for item in os.listdir(desktop_path):
            full_path = os.path.join(desktop_path, item)
            if os.path.isdir(full_path):
                items.append(f"📁 [ПАПКА] {item}")
            else:
                size = os.path.getsize(full_path)
                size_str = f"{size / (1024*1024):.1f} МБ" if size > 1024*1024 else f"{size / 1024:.1f} КБ" if size > 1024 else f"{size} Б"
                items.append(f"📄 [ФАЙЛ] {item} ({size_str})")
        if not items: return "На рабочем столе пусто."
        return f"=== СОДЕРЖИМОЕ РАБОЧЕГО СТОЛА ({len(items)} объектов) ===\n" + "\n".join(sorted(items))
    except Exception as e:
        return f"Ошибка чтения рабочего стола: {e}"

def read_document(file_path: str) -> str:
    if not os.path.exists(file_path): return f"Файл не найден: {file_path}"
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(file_path, dpi=200)
            all_text = []
            for i, img in enumerate(images):
                temp_path = os.path.join(WORK_DIR, f"_doc_page_{i}.jpg")
                img.save(temp_path, 'JPEG')
                all_text.append(f"Страница {i+1}:\n{analyze_image(temp_path, 'Извлеки весь текст с этой страницы')}")
                os.remove(temp_path)
            return "\n\n".join(all_text)
        except ImportError:
            return "Для чтения PDF установите: pip install pdf2image poppler-utils"
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
        return analyze_image(file_path, "Извлеки весь текст с этого изображения")
    return f"Неподдерживаемый формат: {ext}"

def read_table(file_path: str) -> str:
    if not os.path.exists(file_path): return f"Файл не найден: {file_path}"
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        try:
            import pandas as pd
            return pd.read_csv(file_path).to_string(index=False)
        except ImportError: return "Установите pandas: pip install pandas"
    elif ext in ['.xlsx', '.xls']:
        try:
            import pandas as pd
            return pd.read_excel(file_path).to_string(index=False)
        except ImportError: return "Установите openpyxl: pip install openpyxl"
    return f"Неподдерживаемый формат таблицы: {ext}"

def deep_research(query: str, depth: str = "medium") -> str:
    depth_map = {"fast": {"search": 20, "sites": 15}, "medium": {"search": 40, "sites": 30}, "deep": {"search": 70, "sites": 50}}
    cfg = depth_map.get(depth, depth_map["medium"])
    print(f"🔍 Исследование ({depth}): ищу {cfg['search']} ссылок...")
    try:
        results = list(DDGS().text(query, region="wt-wt", max_results=cfg["search"]))
    except Exception as e:
        return f"Ошибка поиска: {e}"
    if not results: return "Ничего не найдено."
    urls = [r["href"] for r in results[:cfg["sites"]]]
    print(f"🌐 Загружаю {len(urls)} сайтов...")
    fetched = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(_fetch_single_site, url): url for url in urls}
        for future in as_completed(future_to_url):
            result = future.result()
            if result["success"]: fetched.append(result)
    if not fetched: return "Не удалось загрузить ни один сайт."
    print(f"📚 Собрано {len(fetched)} источников")
    output = [f"=== ИССЛЕДОВАНИЕ: {query} ===", f"Источников: {len(fetched)}", ""]
    for i, site in enumerate(fetched, 1):
        output.extend([f"--- ИСТОЧНИК {i} ---", f"Заголовок: {site['title']}", f"URL: {site['url']}", f"Содержимое:\n{site['content'][:2500]}", ""])
    output.append("=== Используй ТОЛЬКО информацию из этих источников ===")
    return "\n".join(output)

def _fetch_single_site(url: str, timeout: int = 12) -> dict:
    result = {"url": url, "title": "", "content": "", "success": False}
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', 'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8'}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.encoding = response.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('title')
        result["title"] = title_tag.get_text(strip=True) if title_tag else url
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']): tag.decompose()
        main = soup.select_one('article') or soup.select_one('main') or soup.body
        lines = [l.strip() for l in main.get_text(separator='\n').split('\n') if len(l.strip()) > 30]
        result["content"] = '\n'.join(lines)[:4000]
        result["success"] = len(result["content"]) > 100
    except: pass
    return result


# ================= 🎯 МАКСИМАЛЬНЫЙ СЛОВАРЬ ПРИЛОЖЕНИЙ =================

STANDARD_APPS = {
    # ============ 🖥️ СИСТЕМНЫЕ ПРИЛОЖЕНИЯ ============
    "блокнот": [r"C:\Windows\System32\notepad.exe", "notepad"],
    "notepad": [r"C:\Windows\System32\notepad.exe", "notepad"],
    "калькулятор": ["calc"],
    "calculator": ["calc"],
    "проводник": ["explorer"],
    "explorer": ["explorer"],
    "мой компьютер": ["explorer"],
    "этот компьютер": ["explorer"],
    "paint": ["mspaint", r"C:\Windows\System32\mspaint.exe"],
    "пэинт": ["mspaint"],
    "командная строка": ["cmd"],
    "cmd": ["cmd"],
    "терминал": ["powershell", "wt"],
    "powershell": ["powershell"],
    "windows terminal": ["wt"],
    "диспетчер задач": ["taskmgr"],
    "task manager": ["taskmgr"],
    "настройки": ["start ms-settings:"],
    "settings": ["start ms-settings:"],
    "панель управления": ["control"],
    "control panel": ["control"],
    "реестр": ["regedit"],
    "regedit": ["regedit"],
    "службы": ["services.msc"],
    "services": ["services.msc"],
    "устройства": ["devmgmt.msc"],
    "device manager": ["devmgmt.msc"],
    "диск": ["diskmgmt.msc"],
    "disk management": ["diskmgmt.msc"],
    "очистка диска": ["cleanmgr"],
    "wordpad": ["wordpad"],
    "ножницы": ["snippingtool"],
    "snipping tool": ["snippingtool"],
    "скриншот": ["snippingtool"],
    "запись экрана": ["start ms-screenclip:"],
    "часы": ["start ms-clock:"],
    "будильник": ["start ms-clock:"],
    "камера": ["start microsoft.windows.camera:"],
    "webcam": ["start microsoft.windows.camera:"],
    "веб-камера": ["start microsoft.windows.camera:"],
    "погода": ["start ms-weather:"],
    "weather": ["start ms-weather:"],
    "карты": ["start bingmaps:"],
    "maps": ["start bingmaps:"],
    "магазин": ["start ms-windows-store:"],
    "microsoft store": ["start ms-windows-store:"],
    "windows store": ["start ms-windows-store:"],
    "xbox": ["start ms-xbox:"],
    "игровая панель": ["start ms-gamebar:"],
    "game bar": ["start ms-gamebar:"],
    
    # ============ 🌐 БРАУЗЕРЫ ============
    "chrome": [
        f"{PROGRAM_FILES}\\Google\\Chrome\\Application\\chrome.exe",
        f"{PROGRAM_FILES_X86}\\Google\\Chrome\\Application\\chrome.exe",
        f"{LOCALAPPDATA}\\Google\\Chrome\\Application\\chrome.exe",
    ],
    "google chrome": [
        f"{PROGRAM_FILES}\\Google\\Chrome\\Application\\chrome.exe",
        f"{LOCALAPPDATA}\\Google\\Chrome\\Application\\chrome.exe",
    ],
    "хром": [f"{LOCALAPPDATA}\\Google\\Chrome\\Application\\chrome.exe"],
    "firefox": [
        f"{PROGRAM_FILES}\\Mozilla Firefox\\firefox.exe",
        f"{PROGRAM_FILES_X86}\\Mozilla Firefox\\firefox.exe",
    ],
    "фаерфокс": [f"{PROGRAM_FILES}\\Mozilla Firefox\\firefox.exe"],
    "edge": [f"{PROGRAM_FILES_X86}\\Microsoft\\Edge\\Application\\msedge.exe"],
    "microsoft edge": [f"{PROGRAM_FILES_X86}\\Microsoft\\Edge\\Application\\msedge.exe"],
    "эдж": [f"{PROGRAM_FILES_X86}\\Microsoft\\Edge\\Application\\msedge.exe"],
    "яндекс браузер": [
        f"{LOCALAPPDATA}\\Yandex\\YandexBrowser\\Application\\browser.exe",
        f"{PROGRAM_FILES}\\Yandex\\YandexBrowser\\Application\\browser.exe",
    ],
    "yandex browser": [f"{LOCALAPPDATA}\\Yandex\\YandexBrowser\\Application\\browser.exe"],
    "яндекс": [f"{LOCALAPPDATA}\\Yandex\\YandexBrowser\\Application\\browser.exe", "https://ya.ru"],
    "opera": [
        f"{LOCALAPPDATA}\\Programs\\Opera\\opera.exe",
        f"{LOCALAPPDATA}\\Opera Software\\Opera Stable\\opera.exe",
    ],
    "опера": [f"{LOCALAPPDATA}\\Programs\\Opera\\opera.exe"],
    "opera gx": [f"{LOCALAPPDATA}\\Programs\\Opera GX\\opera.exe"],
    "brave": [
        f"{PROGRAM_FILES}\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
        f"{LOCALAPPDATA}\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
    ],
    "brave browser": [f"{PROGRAM_FILES}\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"],
    "vivaldi": [f"{LOCALAPPDATA}\\Vivaldi\\Application\\vivaldi.exe"],
    "вивальди": [f"{LOCALAPPDATA}\\Vivaldi\\Application\\vivaldi.exe"],
    "tor": [f"{PROGRAM_FILES}\\Tor Browser\\Browser\\firefox.exe"],
    "tor browser": [f"{PROGRAM_FILES}\\Tor Browser\\Browser\\firefox.exe"],
    "safari": ["https://www.apple.com/safari/"],
    "internet explorer": [f"{PROGRAM_FILES}\\Internet Explorer\\iexplore.exe"],
    "ie": [f"{PROGRAM_FILES}\\Internet Explorer\\iexplore.exe"],
    
    # ============ 💬 МЕССЕНДЖЕРЫ ============
    "telegram": [
        f"{APPDATA}\\Telegram Desktop\\Telegram.exe",
        f"{LOCALAPPDATA}\\Telegram Desktop\\Telegram.exe",
        f"{PROGRAM_FILES}\\Telegram Desktop\\Telegram.exe",
        "https://web.telegram.org/",
    ],
    "телеграм": [f"{APPDATA}\\Telegram Desktop\\Telegram.exe", "https://web.telegram.org/"],
    "тг": [f"{APPDATA}\\Telegram Desktop\\Telegram.exe", "https://web.telegram.org/"],
    "discord": [
        f"{LOCALAPPDATA}\\Discord\\Update.exe",
        f"{APPDATA}\\discord\\Discord.exe",
        "https://discord.com/app",
    ],
    "дискорд": [f"{LOCALAPPDATA}\\Discord\\Update.exe", "https://discord.com/app"],
    "whatsapp": [
        f"{LOCALAPPDATA}\\Microsoft\\WindowsApps\\WhatsApp.exe",
        "https://web.whatsapp.com/",
    ],
    "ватсап": [f"{LOCALAPPDATA}\\Microsoft\\WindowsApps\\WhatsApp.exe", "https://web.whatsapp.com/"],
    "вацап": ["https://web.whatsapp.com/"],
    "viber": [
        f"{LOCALAPPDATA}\\Viber\\Viber.exe",
        f"{PROGRAM_FILES}\\Viber\\Viber.exe",
    ],
    "вайбер": [f"{LOCALAPPDATA}\\Viber\\Viber.exe"],
    "skype": [
        f"{LOCALAPPDATA}\\Microsoft\\WindowsApps\\Skype.exe",
        f"{PROGRAM_FILES_X86}\\Microsoft\\Skype for Desktop\\Skype.exe",
        "https://web.skype.com/",
    ],
    "скайп": [f"{LOCALAPPDATA}\\Microsoft\\WindowsApps\\Skype.exe", "https://web.skype.com/"],
    "icq": [f"{LOCALAPPDATA}\\ICQ\\icq.exe"],
    "аська": [f"{LOCALAPPDATA}\\ICQ\\icq.exe"],
    "signal": [f"{LOCALAPPDATA}\\Programs\\signal-desktop\\Signal.exe"],
    "сигнал": [f"{LOCALAPPDATA}\\Programs\\signal-desktop\\Signal.exe"],
    "slack": [
        f"{LOCALAPPDATA}\\slack\\slack.exe",
        f"{APPDATA}\\Slack\\slack.exe",
        "https://app.slack.com/",
    ],
    "слэк": [f"{LOCALAPPDATA}\\slack\\slack.exe", "https://app.slack.com/"],
    "teams": [
        f"{LOCALAPPDATA}\\Microsoft\\Teams\\current\\Teams.exe",
        f"{LOCALAPPDATA}\\Microsoft\\Teams\\Update.exe",
        "https://teams.microsoft.com/",
    ],
    "microsoft teams": [f"{LOCALAPPDATA}\\Microsoft\\Teams\\current\\Teams.exe"],
    "тимс": ["https://teams.microsoft.com/"],
    "zoom": [
        f"{APPDATA}\\Zoom\\bin\\Zoom.exe",
        f"{PROGRAM_FILES}\\Zoom\\bin\\Zoom.exe",
        "https://zoom.us/",
    ],
    "зум": [f"{APPDATA}\\Zoom\\bin\\Zoom.exe", "https://zoom.us/"],
    "zoo": [f"{APPDATA}\\Zoom\\bin\\Zoom.exe"],
    "google meet": ["https://meet.google.com/"],
    "гугл мит": ["https://meet.google.com/"],
    "meet": ["https://meet.google.com/"],
    
    # ============ 📱 СОЦСЕТИ ============
    "vk": ["https://vk.com/"],
    "вк": ["https://vk.com/"],
    "вконтакте": ["https://vk.com/"],
    "vk.com": ["https://vk.com/"],
    "одноклассники": ["https://ok.ru/"],
    "ок": ["https://ok.ru/"],
    "ok.ru": ["https://ok.ru/"],
    "facebook": ["https://facebook.com/"],
    "фейсбук": ["https://facebook.com/"],
    "fb": ["https://facebook.com/"],
    "instagram": ["https://instagram.com/"],
    "инстаграм": ["https://instagram.com/"],
    "инста": ["https://instagram.com/"],
    "ig": ["https://instagram.com/"],
    "twitter": ["https://twitter.com/"],
    "твиттер": ["https://twitter.com/"],
    "x": ["https://x.com/"],
    "tiktok": ["https://www.tiktok.com/"],
    "тикток": ["https://www.tiktok.com/"],
    "pinterest": ["https://pinterest.com/"],
    "пинтерест": ["https://pinterest.com/"],
    "reddit": ["https://reddit.com/"],
    "реддит": ["https://reddit.com/"],
    "linkedin": ["https://linkedin.com/"],
    "линкедин": ["https://linkedin.com/"],
    "twitch": ["https://twitch.tv/"],
    "твич": ["https://twitch.tv/"],
    "tumblr": ["https://tumblr.com/"],
    "тамблер": ["https://tumblr.com/"],
    "snapchat": ["https://snapchat.com/"],
    "снэпчат": ["https://snapchat.com/"],
    
    # ============ 🎮 ИГРОВЫЕ ЛАУНЧЕРЫ ============
    "steam": [
        f"{PROGRAM_FILES_X86}\\Steam\\steam.exe",
        f"{PROGRAM_FILES}\\Steam\\steam.exe",
        r"D:\Steam\steam.exe",
        r"D:\Program Files (x86)\Steam\steam.exe",
        r"E:\Steam\steam.exe",
        "https://store.steampowered.com/",
    ],
    "стим": [f"{PROGRAM_FILES_X86}\\Steam\\steam.exe", "https://store.steampowered.com/"],
    "epic games": [
        f"{PROGRAM_FILES_X86}\\Epic Games\\Launcher\\Portal\\Binaries\\Win32\\EpicGamesLauncher.exe",
        f"{PROGRAM_FILES}\\Epic Games\\Launcher\\Portal\\Binaries\\Win64\\EpicGamesLauncher.exe",
    ],
    "epic games launcher": [f"{PROGRAM_FILES_X86}\\Epic Games\\Launcher\\Portal\\Binaries\\Win32\\EpicGamesLauncher.exe"],
    "epic": [f"{PROGRAM_FILES_X86}\\Epic Games\\Launcher\\Portal\\Binaries\\Win32\\EpicGamesLauncher.exe"],
    "эпик": [f"{PROGRAM_FILES_X86}\\Epic Games\\Launcher\\Portal\\Binaries\\Win32\\EpicGamesLauncher.exe"],
    "battle.net": [
        f"{PROGRAM_FILES_X86}\\Battle.net\\Battle.net Launcher.exe",
        f"{PROGRAM_FILES}\\Battle.net\\Battle.net Launcher.exe",
    ],
    "батлнет": [f"{PROGRAM_FILES_X86}\\Battle.net\\Battle.net Launcher.exe"],
    "blizzard": [f"{PROGRAM_FILES_X86}\\Battle.net\\Battle.net Launcher.exe"],
    "близзард": [f"{PROGRAM_FILES_X86}\\Battle.net\\Battle.net Launcher.exe"],
    "gog galaxy": [
        f"{LOCALAPPDATA}\\GOG.com\\Galaxy\\GalaxyClient.exe",
        f"{PROGRAM_FILES_X86}\\GOG Galaxy\\GalaxyClient.exe",
    ],
    "gog": [f"{LOCALAPPDATA}\\GOG.com\\Galaxy\\GalaxyClient.exe"],
    "ubisoft connect": [
        f"{PROGRAM_FILES_X86}\\Ubisoft\\Ubisoft Game Launcher\\upc.exe",
        f"{PROGRAM_FILES}\\Ubisoft\\Ubisoft Game Launcher\\upc.exe",
    ],
    "ubisoft": [f"{PROGRAM_FILES_X86}\\Ubisoft\\Ubisoft Game Launcher\\upc.exe"],
    "юбисофт": [f"{PROGRAM_FILES_X86}\\Ubisoft\\Ubisoft Game Launcher\\upc.exe"],
    "uplay": [f"{PROGRAM_FILES_X86}\\Ubisoft\\Ubisoft Game Launcher\\upc.exe"],
    "ea app": [
        f"{PROGRAM_FILES}\\Electronic Arts\\EA Desktop\\EA Desktop\\EADesktop.exe",
        f"{PROGRAM_FILES}\\EA\\EA\\EA.exe",
    ],
    "ea desktop": [f"{PROGRAM_FILES}\\Electronic Arts\\EA Desktop\\EA Desktop\\EADesktop.exe"],
    "ea": [f"{PROGRAM_FILES}\\Electronic Arts\\EA Desktop\\EA Desktop\\EADesktop.exe"],
    "origin": [f"{PROGRAM_FILES_X86}\\Origin\\Origin.exe"],
    "ориджин": [f"{PROGRAM_FILES_X86}\\Origin\\Origin.exe"],
    "rockstar": [
        f"{PROGRAM_FILES}\\Rockstar Games\\Launcher\\Launcher.exe",
        f"{PROGRAM_FILES_X86}\\Rockstar Games\\Launcher\\Launcher.exe",
    ],
    "rockstar launcher": [f"{PROGRAM_FILES}\\Rockstar Games\\Launcher\\Launcher.exe"],
    "r*": [f"{PROGRAM_FILES}\\Rockstar Games\\Launcher\\Launcher.exe"],
    "gta": [f"{PROGRAM_FILES}\\Rockstar Games\\Launcher\\Launcher.exe"],
    "gog.com": ["https://www.gog.com/"],
    "itch.io": ["https://itch.io/"],
    "itch": ["https://itch.io/"],
    "riot games": [
        f"{LOCALAPPDATA}\\Riot Games\\Riot Client\\RiotClientServices.exe",
        f"{PROGRAM_FILES}\\Riot Vanguard\\RiotClientServices.exe",
    ],
    "riot": [f"{LOCALAPPDATA}\\Riot Games\\Riot Client\\RiotClientServices.exe"],
    "league of legends": [f"{LOCALAPPDATA}\\Riot Games\\Riot Client\\RiotClientServices.exe"],
    "valorant": [f"{LOCALAPPDATA}\\Riot Games\\Riot Client\\RiotClientServices.exe"],
    "варик": [f"{LOCALAPPDATA}\\Riot Games\\Riot Client\\RiotClientServices.exe"],
    "minecraft launcher": [
        f"{LOCALAPPDATA}\\Programs\\Minecraft Launcher\\MinecraftLauncher.exe",
        f"{PROGRAM_FILES_X86}\\Minecraft Launcher\\MinecraftLauncher.exe",
    ],
    "minecraft": [f"{LOCALAPPDATA}\\Programs\\Minecraft Launcher\\MinecraftLauncher.exe"],
    "майнкрафт": [f"{LOCALAPPDATA}\\Programs\\Minecraft Launcher\\MinecraftLauncher.exe"],
    "майн": [f"{LOCALAPPDATA}\\Programs\\Minecraft Launcher\\MinecraftLauncher.exe"],
    "roblox": [
        f"{LOCALAPPDATA}\\Roblox\\Versions\\*\\RobloxPlayerBeta.exe",
        "https://www.roblox.com/",
    ],
    "роблокс": [f"{LOCALAPPDATA}\\Roblox\\Versions\\*\\RobloxPlayerBeta.exe"],
    "genshin impact": [
        f"{PROGRAM_FILES}\\Genshin Impact\\launcher.exe",
        f"{PROGRAM_FILES}\\miHoYo Launcher\\launcher.exe",
    ],
    "геншин": [f"{PROGRAM_FILES}\\Genshin Impact\\launcher.exe"],
    "honkai": [f"{PROGRAM_FILES}\\miHoYo Launcher\\launcher.exe"],
    "war thunder": [
        f"{PROGRAM_FILES}\\WarThunder\\launcher.exe",
        f"{PROGRAM_FILES_X86}\\Steam\\steamapps\\common\\WarThunder\\launcher.exe",
    ],
    "warthunder": [f"{PROGRAM_FILES}\\WarThunder\\launcher.exe"],
    
    # ============ 💻 РАЗРАБОТКА / IDE ============
    "vs code": [
        f"{LOCALAPPDATA}\\Programs\\Microsoft VS Code\\Code.exe",
        f"{PROGRAM_FILES}\\Microsoft VS Code\\Code.exe",
        "code",
    ],
    "visual studio code": [f"{LOCALAPPDATA}\\Programs\\Microsoft VS Code\\Code.exe", "code"],
    "vscode": [f"{LOCALAPPDATA}\\Programs\\Microsoft VS Code\\Code.exe", "code"],
    "вс код": [f"{LOCALAPPDATA}\\Programs\\Microsoft VS Code\\Code.exe", "code"],
    "visual studio": [
        f"{PROGRAM_FILES}\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\devenv.exe",
        f"{PROGRAM_FILES}\\Microsoft Visual Studio\\2022\\Professional\\Common7\\IDE\\devenv.exe",
        f"{PROGRAM_FILES}\\Microsoft Visual Studio\\2022\\Enterprise\\Common7\\IDE\\devenv.exe",
        f"{PROGRAM_FILES}\\Microsoft Visual Studio\\2019\\Community\\Common7\\IDE\\devenv.exe",
        f"{PROGRAM_FILES}\\Microsoft Visual Studio\\2019\\Professional\\Common7\\IDE\\devenv.exe",
        f"{PROGRAM_FILES_X86}\\Microsoft Visual Studio\\2019\\Community\\Common7\\IDE\\devenv.exe",
        f"{PROGRAM_FILES_X86}\\Microsoft Visual Studio\\2017\\Community\\Common7\\IDE\\devenv.exe",
    ],
    "студия": [f"{PROGRAM_FILES}\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\devenv.exe"],
    "visual studio installer": [
        f"{PROGRAM_FILES_X86}\\Microsoft Visual Studio\\Installer\\setup.exe",
    ],
    "pycharm": [
        f"{PROGRAM_FILES}\\JetBrains\\PyCharm*\\bin\\pycharm64.exe",
        f"{LOCALAPPDATA}\\Programs\\PyCharm*\\bin\\pycharm64.exe",
        f"{LOCALAPPDATA}\\JetBrains\\Toolbox\\apps\\PyCharm*\\bin\\pycharm64.exe",
    ],
    "пайчарм": [f"{PROGRAM_FILES}\\JetBrains\\PyCharm*\\bin\\pycharm64.exe"],
    "intellij": [
        f"{PROGRAM_FILES}\\JetBrains\\IntelliJ*\\bin\\idea64.exe",
        f"{LOCALAPPDATA}\\JetBrains\\Toolbox\\apps\\IDEA*\\bin\\idea64.exe",
    ],
    "intellij idea": [f"{PROGRAM_FILES}\\JetBrains\\IntelliJ*\\bin\\idea64.exe"],
    "идея": [f"{PROGRAM_FILES}\\JetBrains\\IntelliJ*\\bin\\idea64.exe"],
    "webstorm": [f"{PROGRAM_FILES}\\JetBrains\\WebStorm*\\bin\\webstorm64.exe"],
    "вебшторм": [f"{PROGRAM_FILES}\\JetBrains\\WebStorm*\\bin\\webstorm64.exe"],
    "phpstorm": [f"{PROGRAM_FILES}\\JetBrains\\PhpStorm*\\bin\\phpstorm64.exe"],
    "пхпшторм": [f"{PROGRAM_FILES}\\JetBrains\\PhpStorm*\\bin\\phpstorm64.exe"],
    "goland": [f"{PROGRAM_FILES}\\JetBrains\\GoLand*\\bin\\goland64.exe"],
    "rider": [f"{PROGRAM_FILES}\\JetBrains\\JetBrains Rider*\\bin\\rider64.exe"],
    "rust rover": [f"{PROGRAM_FILES}\\JetBrains\\RustRover*\\bin\\rustrover64.exe"],
    "clion": [f"{PROGRAM_FILES}\\JetBrains\\CLion*\\bin\\clion64.exe"],
    "datagrip": [f"{PROGRAM_FILES}\\JetBrains\\DataGrip*\\bin\\datagrip64.exe"],
    "android studio": [
        f"{PROGRAM_FILES}\\Android\\Android Studio\\bin\\studio64.exe",
        f"{LOCALAPPDATA}\\Android\\android-studio\\bin\\studio64.exe",
    ],
    "андроид студио": [f"{PROGRAM_FILES}\\Android\\Android Studio\\bin\\studio64.exe"],
    "xcode": ["https://developer.apple.com/xcode/"],
    "eclipse": [
        f"{PROGRAM_FILES}\\Eclipse\\eclipse.exe",
        f"{PROGRAM_FILES_X86}\\Eclipse\\eclipse.exe",
    ],
    "эклипс": [f"{PROGRAM_FILES}\\Eclipse\\eclipse.exe"],
    "netbeans": [f"{PROGRAM_FILES}\\NetBeans*\\bin\\netbeans64.exe"],
    "sublime": [
        f"{PROGRAM_FILES}\\Sublime Text\\sublime_text.exe",
        f"{PROGRAM_FILES}\\Sublime Text 3\\sublime_text.exe",
    ],
    "sublime text": [f"{PROGRAM_FILES}\\Sublime Text\\sublime_text.exe"],
    "сублайм": [f"{PROGRAM_FILES}\\Sublime Text\\sublime_text.exe"],
    "notepad++": [
        f"{PROGRAM_FILES}\\Notepad++\\notepad++.exe",
        f"{PROGRAM_FILES_X86}\\Notepad++\\notepad++.exe",
    ],
    "нотпад++": [f"{PROGRAM_FILES}\\Notepad++\\notepad++.exe"],
    "npp": [f"{PROGRAM_FILES}\\Notepad++\\notepad++.exe"],
    "atom": [f"{LOCALAPPDATA}\\atom\\atom.exe"],
    "brackets": [f"{PROGRAM_FILES}\\Brackets\\Brackets.exe"],
    "vscodium": [f"{LOCALAPPDATA}\\Programs\\VSCodium\\VSCodium.exe"],
    "cursor": [f"{LOCALAPPDATA}\\Programs\\cursor\\Cursor.exe"],
    "курсор": [f"{LOCALAPPDATA}\\Programs\\cursor\\Cursor.exe"],
    "neovim": [f"{PROGRAM_FILES}\\Neovim\\bin\\nvim.exe"],
    "vim": [f"{PROGRAM_FILES}\\Vim\\vim*\\vim.exe"],
    "emacs": [f"{PROGRAM_FILES}\\Emacs\\bin\\emacs.exe"],
    
    # ============ 🎨 ГРАФИКА И ДИЗАЙН ============
    "photoshop": [
        f"{PROGRAM_FILES}\\Adobe\\Adobe Photoshop*\\Photoshop.exe",
        f"{PROGRAM_FILES}\\Adobe\\Adobe Photoshop 2024\\Photoshop.exe",
        f"{PROGRAM_FILES}\\Adobe\\Adobe Photoshop 2023\\Photoshop.exe",
    ],
    "фотошоп": [f"{PROGRAM_FILES}\\Adobe\\Adobe Photoshop*\\Photoshop.exe"],
    "ps": [f"{PROGRAM_FILES}\\Adobe\\Adobe Photoshop*\\Photoshop.exe"],
    "illustrator": [f"{PROGRAM_FILES}\\Adobe\\Adobe Illustrator*\\Support Files\\Contents\\Windows\\Illustrator.exe"],
    "иллюстратор": [f"{PROGRAM_FILES}\\Adobe\\Adobe Illustrator*\\Support Files\\Contents\\Windows\\Illustrator.exe"],
    "премьер": [f"{PROGRAM_FILES}\\Adobe\\Adobe Premiere Pro*\\Adobe Premiere Pro.exe"],
    "premiere": [f"{PROGRAM_FILES}\\Adobe\\Adobe Premiere Pro*\\Adobe Premiere Pro.exe"],
    "premiere pro": [f"{PROGRAM_FILES}\\Adobe\\Adobe Premiere Pro*\\Adobe Premiere Pro.exe"],
    "after effects": [f"{PROGRAM_FILES}\\Adobe\\Adobe After Effects*\\Support Files\\AfterFX.exe"],
    "after": [f"{PROGRAM_FILES}\\Adobe\\Adobe After Effects*\\Support Files\\AfterFX.exe"],
    "афтер": [f"{PROGRAM_FILES}\\Adobe\\Adobe After Effects*\\Support Files\\AfterFX.exe"],
    "indesign": [f"{PROGRAM_FILES}\\Adobe\\Adobe InDesign*\\InDesign.exe"],
    "lightroom": [
        f"{PROGRAM_FILES}\\Adobe\\Adobe Lightroom Classic\\lightroom.exe",
        f"{PROGRAM_FILES}\\Adobe\\Adobe Lightroom\\lightroom.exe",
    ],
    "лайтрум": [f"{PROGRAM_FILES}\\Adobe\\Adobe Lightroom Classic\\lightroom.exe"],
    "xd": [f"{PROGRAM_FILES}\\Adobe\\Adobe XD\\XD.exe"],
    "figma": [
        f"{LOCALAPPDATA}\\Figma\\Figma.exe",
        f"{APPDATA}\\Figma\\Figma.exe",
        "https://www.figma.com/",
    ],
    "фигма": [f"{LOCALAPPDATA}\\Figma\\Figma.exe", "https://www.figma.com/"],
    "sketch": ["https://www.sketch.com/"],
    "gimp": [
        f"{PROGRAM_FILES}\\GIMP*\\bin\\gimp-*.exe",
        f"{PROGRAM_FILES}\\GIMP 2\\bin\\gimp-2.10.exe",
    ],
    "гимп": [f"{PROGRAM_FILES}\\GIMP*\\bin\\gimp-*.exe"],
    "inkscape": [f"{PROGRAM_FILES}\\Inkscape\\bin\\inkscape.exe"],
    "инкскейп": [f"{PROGRAM_FILES}\\Inkscape\\bin\\inkscape.exe"],
    "krita": [f"{PROGRAM_FILES}\\Krita\\bin\\krita.exe"],
    "крита": [f"{PROGRAM_FILES}\\Krita\\bin\\krita.exe"],
    "paint.net": [f"{PROGRAM_FILES}\\paint.net\\PaintDotNet.exe"],
    "пейнт нет": [f"{PROGRAM_FILES}\\paint.net\\PaintDotNet.exe"],
    "canva": ["https://www.canva.com/"],
    "канва": ["https://www.canva.com/"],
    "coreldraw": [f"{PROGRAM_FILES}\\Corel\\CorelDRAW*\\Programs\\CorelDRW.exe"],
    "корел": [f"{PROGRAM_FILES}\\Corel\\CorelDRAW*\\Programs\\CorelDRW.exe"],
    "autocad": [
        f"{PROGRAM_FILES}\\Autodesk\\AutoCAD*\\acad.exe",
        f"{PROGRAM_FILES}\\Autodesk\\AutoCAD 2024\\acad.exe",
    ],
    "автокад": [f"{PROGRAM_FILES}\\Autodesk\\AutoCAD*\\acad.exe"],
    "blender": [
        f"{PROGRAM_FILES}\\Blender Foundation\\Blender*\\blender.exe",
        f"{PROGRAM_FILES}\\Blender Foundation\\Blender 4.0\\blender.exe",
    ],
    "блендер": [f"{PROGRAM_FILES}\\Blender Foundation\\Blender*\\blender.exe"],
    "maya": [f"{PROGRAM_FILES}\\Autodesk\\Maya*\\bin\\maya.exe"],
    "майа": [f"{PROGRAM_FILES}\\Autodesk\\Maya*\\bin\\maya.exe"],
    "3ds max": [f"{PROGRAM_FILES}\\Autodesk\\3ds Max*\\3dsmax.exe"],
    "3d max": [f"{PROGRAM_FILES}\\Autodesk\\3ds Max*\\3dsmax.exe"],
    "cinema 4d": [f"{PROGRAM_FILES}\\Maxon Cinema 4D*\\Cinema 4D.exe"],
    "unity": [
        f"{PROGRAM_FILES}\\Unity\\Hub\\Editor\\*\\Editor\\Unity.exe",
        f"{PROGRAM_FILES}\\Unity Hub\\Unity Hub.exe",
    ],
    "unity hub": [f"{PROGRAM_FILES}\\Unity Hub\\Unity Hub.exe"],
    "юнити": [f"{PROGRAM_FILES}\\Unity Hub\\Unity Hub.exe"],
    "unreal engine": [
        f"{PROGRAM_FILES}\\Epic Games\\Launcher\\Portal\\Binaries\\Win32\\EpicGamesLauncher.exe",
        f"{PROGRAM_FILES}\\Epic Games\\UE_*\\Engine\\Binaries\\Win64\\UE4Editor.exe",
    ],
    "unreal": [f"{PROGRAM_FILES}\\Epic Games\\Launcher\\Portal\\Binaries\\Win32\\EpicGamesLauncher.exe"],
    "godot": [f"{LOCALAPPDATA}\\Godot\\Godot*.exe"],
    "годот": [f"{LOCALAPPDATA}\\Godot\\Godot*.exe"],
    
    # ============ 🎵 МУЗЫКА И АУДИО ============
    "spotify": [
        f"{APPDATA}\\Spotify\\Spotify.exe",
        f"{LOCALAPPDATA}\\Microsoft\\WindowsApps\\Spotify.exe",
        "https://open.spotify.com/",
    ],
    "спотифай": [f"{APPDATA}\\Spotify\\Spotify.exe", "https://open.spotify.com/"],
    "яндекс музыка": ["https://music.yandex.ru/"],
    "yandex music": ["https://music.yandex.ru/"],
    "apple music": ["https://music.apple.com/"],
    "itunes": [f"{PROGRAM_FILES}\\iTunes\\iTunes.exe"],
    "аитюнс": [f"{PROGRAM_FILES}\\iTunes\\iTunes.exe"],
    "soundcloud": ["https://soundcloud.com/"],
    "саундклауд": ["https://soundcloud.com/"],
    "deezer": ["https://www.deezer.com/"],
    "tidal": ["https://tidal.com/"],
    "youtube music": ["https://music.youtube.com/"],
    "ютюб мьюзик": ["https://music.youtube.com/"],
    "fl studio": [
        f"{PROGRAM_FILES}\\Image-Line\\FL Studio*\\FL64.exe",
        f"{PROGRAM_FILES_X86}\\Image-Line\\FL Studio*\\FL.exe",
    ],
    "фл студио": [f"{PROGRAM_FILES}\\Image-Line\\FL Studio*\\FL64.exe"],
    "фрукты": [f"{PROGRAM_FILES}\\Image-Line\\FL Studio*\\FL64.exe"],
    "ableton": [f"{PROGRAM_FILES}\\Ableton\\Live*\\Program\\Ableton Live*.exe"],
    "аблетон": [f"{PROGRAM_FILES}\\Ableton\\Live*\\Program\\Ableton Live*.exe"],
    "cubase": [f"{PROGRAM_FILES}\\Steinberg\\Cubase*\\Cubase*.exe"],
    "reason": [f"{PROGRAM_FILES}\\Reason Studios\\Reason*\\Reason.exe"],
    "reaper": [f"{PROGRAM_FILES}\\REAPER*\\reaper.exe"],
    "audacity": [
        f"{PROGRAM_FILES}\\Audacity\\Audacity.exe",
        f"{PROGRAM_FILES_X86}\\Audacity\\Audacity.exe",
    ],
    "аудасити": [f"{PROGRAM_FILES}\\Audacity\\Audacity.exe"],
    "audition": [f"{PROGRAM_FILES}\\Adobe\\Adobe Audition*\\Adobe Audition.exe"],
    "аудишн": [f"{PROGRAM_FILES}\\Adobe\\Adobe Audition*\\Adobe Audition.exe"],
    
    # ============ 🎬 ВИДЕО И СТРИМИНГ ============
    "obs": [
        f"{PROGRAM_FILES}\\obs-studio\\bin\\64bit\\obs64.exe",
        f"{PROGRAM_FILES_X86}\\obs-studio\\bin\\64bit\\obs64.exe",
    ],
    "obs studio": [f"{PROGRAM_FILES}\\obs-studio\\bin\\64bit\\obs64.exe"],
    "обс": [f"{PROGRAM_FILES}\\obs-studio\\bin\\64bit\\obs64.exe"],
    "vlc": [
        f"{PROGRAM_FILES}\\VideoLAN\\VLC\\vlc.exe",
        f"{PROGRAM_FILES_X86}\\VideoLAN\\VLC\\vlc.exe",
    ],
    "влс": [f"{PROGRAM_FILES}\\VideoLAN\\VLC\\vlc.exe"],
    "mpc": [
        f"{PROGRAM_FILES}\\MPC-HC\\mpc-hc64.exe",
        f"{PROGRAM_FILES_X86}\\MPC-HC\\mpc-hc.exe",
    ],
    "mpc-hc": [f"{PROGRAM_FILES}\\MPC-HC\\mpc-hc64.exe"],
    "kmplayer": [f"{PROGRAM_FILES}\\KMPlayer 64X\\KMPlayer64.exe"],
    "potplayer": [f"{PROGRAM_FILES}\\DAUM\\PotPlayer\\PotPlayerMini64.exe"],
    "потплеер": [f"{PROGRAM_FILES}\\DAUM\\PotPlayer\\PotPlayerMini64.exe"],
    "gom player": [f"{PROGRAM_FILES_X86}\\GRETECH\\GomPlayer\\GOM.exe"],
    "davinci": [
        f"{PROGRAM_FILES}\\Blackmagic Design\\DaVinci Resolve\\Resolve.exe",
    ],
    "davinci resolve": [f"{PROGRAM_FILES}\\Blackmagic Design\\DaVinci Resolve\\Resolve.exe"],
    "давинчи": [f"{PROGRAM_FILES}\\Blackmagic Design\\DaVinci Resolve\\Resolve.exe"],
    "filmora": [f"{PROGRAM_FILES_X86}\\Wondershare\\Wondershare Filmora*\\Filmora.exe"],
    "wondershare": [f"{PROGRAM_FILES_X86}\\Wondershare\\Wondershare Filmora*\\Filmora.exe"],
    "capcut": [
        f"{LOCALAPPDATA}\\CapCut\\CapCut.exe",
        "https://www.capcut.com/",
    ],
    "капкат": [f"{LOCALAPPDATA}\\CapCut\\CapCut.exe"],
    "camtasia": [f"{PROGRAM_FILES}\\TechSmith\\Camtasia*\\CamtasiaStudio.exe"],
    "kinemaster": ["https://www.kinemaster.com/"],
    "handbrake": [f"{PROGRAM_FILES}\\HandBrake\\HandBrake.exe"],
    "ffmpeg": ["ffmpeg"],
    "youtube": ["https://www.youtube.com/"],
    "ютуб": ["https://www.youtube.com/"],
    "рутюб": ["https://rutube.ru/"],
    "rutube": ["https://rutube.ru/"],
    "twitch": ["https://www.twitch.tv/"],
    "твич": ["https://www.twitch.tv/"],
    "kinopoisk": ["https://www.kinopoisk.ru/"],
    "кинопоиск": ["https://www.kinopoisk.ru/"],
    "ivi": ["https://www.ivi.ru/"],
    "иви": ["https://www.ivi.ru/"],
    "okko": ["https://okko.tv/"],
    "окко": ["https://okko.tv/"],
    "more.tv": ["https://more.tv/"],
    "моретв": ["https://more.tv/"],
    "premier": ["https://premier.one/"],
    "премьер": ["https://premier.one/"],
    "wink": ["https://wink.ru/"],
    "винк": ["https://wink.ru/"],
    "netflix": ["https://www.netflix.com/"],
    "нетфликс": ["https://www.netflix.com/"],
    "hbo": ["https://www.max.com/"],
    "disney+": ["https://www.disneyplus.com/"],
    "amazon prime": ["https://www.primevideo.com/"],
    "apple tv": ["https://tv.apple.com/"],
    
    # ============ 📝 ОФИС И ДОКУМЕНТЫ ============
    "word": [
        f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
        f"{PROGRAM_FILES_X86}\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
        f"{PROGRAM_FILES}\\Microsoft Office\\Office16\\WINWORD.EXE",
    ],
    "ворд": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\WINWORD.EXE"],
    "microsoft word": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\WINWORD.EXE"],
    "excel": [
        f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\EXCEL.EXE",
        f"{PROGRAM_FILES_X86}\\Microsoft Office\\root\\Office16\\EXCEL.EXE",
    ],
    "эксель": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\EXCEL.EXE"],
    "microsoft excel": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\EXCEL.EXE"],
    "таблица": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\EXCEL.EXE"],
    "таблицу": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\EXCEL.EXE"],
    "powerpoint": [
        f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\POWERPNT.EXE",
        f"{PROGRAM_FILES_X86}\\Microsoft Office\\root\\Office16\\POWERPNT.EXE",
    ],
    "пауэрпоинт": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\POWERPNT.EXE"],
    "презентация": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\POWERPNT.EXE"],
    "outlook": [
        f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\OUTLOOK.EXE",
        f"{PROGRAM_FILES_X86}\\Microsoft Office\\root\\Office16\\OUTLOOK.EXE",
    ],
    "аутлук": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\OUTLOOK.EXE"],
    "почта outlook": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\OUTLOOK.EXE"],
    "access": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\MSACCESS.EXE"],
    "publisher": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\MSPUB.EXE"],
    "onenote": [f"{PROGRAM_FILES}\\Microsoft Office\\root\\Office16\\ONENOTE.EXE"],
    "libreoffice": [
        f"{PROGRAM_FILES}\\LibreOffice\\program\\soffice.exe",
        f"{PROGRAM_FILES_X86}\\LibreOffice\\program\\soffice.exe",
    ],
    "либреофис": [f"{PROGRAM_FILES}\\LibreOffice\\program\\soffice.exe"],
    "openoffice": [f"{PROGRAM_FILES_X86}\\OpenOffice*\\program\\soffice.exe"],
    "writer": [f"{PROGRAM_FILES}\\LibreOffice\\program\\swriter.exe"],
    "calc": [f"{PROGRAM_FILES}\\LibreOffice\\program\\scalc.exe"],
    "google docs": ["https://docs.google.com/"],
    "гугл докс": ["https://docs.google.com/"],
    "google sheets": ["https://sheets.google.com/"],
    "гугл таблицы": ["https://sheets.google.com/"],
    "google slides": ["https://slides.google.com/"],
    "гугл презентации": ["https://slides.google.com/"],
    "google drive": ["https://drive.google.com/"],
    "гугл диск": ["https://drive.google.com/"],
    "notion": [
        f"{LOCALAPPDATA}\\Programs\\Notion\\Notion.exe",
        f"{APPDATA}\\Notion\\Notion.exe",
        "https://www.notion.so/",
    ],
    "ноушн": [f"{LOCALAPPDATA}\\Programs\\Notion\\Notion.exe", "https://www.notion.so/"],
    "obsidian": [
        f"{LOCALAPPDATA}\\Obsidian\\Obsidian.exe",
        f"{APPDATA}\\obsidian\\Obsidian.exe",
    ],
    "обсидиан": [f"{LOCALAPPDATA}\\Obsidian\\Obsidian.exe"],
    "evernote": [f"{LOCALAPPDATA}\\Evernote\\Evernote.exe", "https://evernote.com/"],
    "эверноут": [f"{LOCALAPPDATA}\\Evernote\\Evernote.exe"],
    "adobe acrobat": [
        f"{PROGRAM_FILES}\\Adobe\\Acrobat DC\\Acrobat\\Acrobat.exe",
        f"{PROGRAM_FILES_X86}\\Adobe\\Acrobat Reader DC\\Reader\\AcroRd32.exe",
    ],
    "acrobat": [f"{PROGRAM_FILES}\\Adobe\\Acrobat DC\\Acrobat\\Acrobat.exe"],
    "акробат": [f"{PROGRAM_FILES}\\Adobe\\Acrobat DC\\Acrobat\\Acrobat.exe"],
    "foxit reader": [
        f"{PROGRAM_FILES}\\Foxit Software\\Foxit PDF Reader\\FoxitPDFReader.exe",
        f"{PROGRAM_FILES_X86}\\Foxit Software\\Foxit PDF Reader\\FoxitPDFReader.exe",
    ],
    "foxit": [f"{PROGRAM_FILES}\\Foxit Software\\Foxit PDF Reader\\FoxitPDFReader.exe"],
    "sumatra": [f"{PROGRAM_FILES}\\SumatraPDF\\SumatraPDF.exe"],
    "sumatrapdf": [f"{PROGRAM_FILES}\\SumatraPDF\\SumatraPDF.exe"],
    
    # ============ 📦 АРХИВАТОРЫ ============
    "winrar": [
        f"{PROGRAM_FILES}\\WinRAR\\WinRAR.exe",
        f"{PROGRAM_FILES_X86}\\WinRAR\\WinRAR.exe",
    ],
    "винрар": [f"{PROGRAM_FILES}\\WinRAR\\WinRAR.exe"],
    "rar": [f"{PROGRAM_FILES}\\WinRAR\\WinRAR.exe"],
    "7zip": [
        f"{PROGRAM_FILES}\\7-Zip\\7zFM.exe",
        f"{PROGRAM_FILES_X86}\\7-Zip\\7zFM.exe",
    ],
    "7-zip": [f"{PROGRAM_FILES}\\7-Zip\\7zFM.exe"],
    "семь зип": [f"{PROGRAM_FILES}\\7-Zip\\7zFM.exe"],
    "winzip": [f"{PROGRAM_FILES_X86}\\WinZip\\WINZIP32.EXE"],
    "peazip": [f"{PROGRAM_FILES}\\PeaZip\\peazip.exe"],
    
    # ============ 💾 ФАЙЛОВЫЕ МЕНЕДЖЕРЫ ============
    "total commander": [
        f"{PROGRAM_FILES}\\totalcmd\\TOTALCMD64.EXE",
        f"{PROGRAM_FILES_X86}\\totalcmd\\TOTALCMD.EXE",
    ],
    "тотал командер": [f"{PROGRAM_FILES}\\totalcmd\\TOTALCMD64.EXE"],
    "total": [f"{PROGRAM_FILES}\\totalcmd\\TOTALCMD64.EXE"],
    "far manager": [f"{PROGRAM_FILES}\\Far Manager\\Far.exe"],
    "far": [f"{PROGRAM_FILES}\\Far Manager\\Far.exe"],
    "файр": [f"{PROGRAM_FILES}\\Far Manager\\Far.exe"],
    "files": ["start ms-files:"],
    "double commander": [f"{PROGRAM_FILES}\\Double Commander\\doublecmd.exe"],
    "free commander": [f"{PROGRAM_FILES_X86}\\FreeCommander*\\freecommander.exe"],
    
    # ============ 📥 ТОРРЕНТЫ И ЗАГРУЗКИ ============
    "utorrent": [
        f"{APPDATA}\\uTorrent\\uTorrent.exe",
        f"{LOCALAPPDATA}\\uTorrent\\uTorrent.exe",
    ],
    "utorrent web": ["https://www.utorrent.com/"],
    "utoru": [f"{APPDATA}\\uTorrent\\uTorrent.exe"],
    "bittorrent": [f"{APPDATA}\\BitTorrent\\BitTorrent.exe"],
    "qbit": [
        f"{PROGRAM_FILES}\\qBittorrent\\qbittorrent.exe",
        f"{PROGRAM_FILES_X86}\\qBittorrent\\qbittorrent.exe",
    ],
    "qbittorrent": [f"{PROGRAM_FILES}\\qBittorrent\\qbittorrent.exe"],
    "кубит": [f"{PROGRAM_FILES}\\qBittorrent\\qbittorrent.exe"],
    "transmission": [f"{PROGRAM_FILES}\\Transmission\\transmission-qt.exe"],
    "deluge": [f"{PROGRAM_FILES}\\Deluge\\deluge.exe"],
    "torrent": [f"{PROGRAM_FILES}\\qBittorrent\\qbittorrent.exe"],
    "торрент": [f"{PROGRAM_FILES}\\qBittorrent\\qbittorrent.exe"],
    "idm": [
        f"{PROGRAM_FILES_X86}\\Internet Download Manager\\IDMan.exe",
        f"{PROGRAM_FILES}\\Internet Download Manager\\IDMan.exe",
    ],
    "internet download manager": [f"{PROGRAM_FILES_X86}\\Internet Download Manager\\IDMan.exe"],
    "download master": [f"{LOCALAPPDATA}\\DownloadMaster\\dm.exe"],
    "free download manager": [
        f"{PROGRAM_FILES}\\Free Download Manager\\fdm.exe",
        f"{LOCALAPPDATA}\\Free Download Manager\\fdm.exe",
    ],
    
    # ============ 🔐 VPN И БЕЗОПАСНОСТЬ ============
    "nordvpn": [
        f"{PROGRAM_FILES}\\NordVPN\\NordVPN.exe",
        f"{LOCALAPPDATA}\\NordVPN\\NordVPN.exe",
    ],
    "нордвпн": [f"{PROGRAM_FILES}\\NordVPN\\NordVPN.exe"],
    "expressvpn": [f"{PROGRAM_FILES_X86}\\ExpressVPN\\ExpressVPN.exe"],
    "express vpn": [f"{PROGRAM_FILES_X86}\\ExpressVPN\\ExpressVPN.exe"],
    "protonvpn": [
        f"{LOCALAPPDATA}\\Programs\\protonvpn\\ProtonVPN.exe",
        f"{PROGRAM_FILES}\\Proton\\VPN\\ProtonVPN.exe",
    ],
    "proton vpn": [f"{LOCALAPPDATA}\\Programs\\protonvpn\\ProtonVPN.exe"],
    "surfshark": [f"{LOCALAPPDATA}\\Programs\\Surfshark\\Surfshark.exe"],
    "kaspersky": [
        f"{PROGRAM_FILES}\\Kaspersky Lab\\Kaspersky*\\avp.exe",
        f"{PROGRAM_FILES_X86}\\Kaspersky Lab\\Kaspersky*\\avp.exe",
    ],
    "касперский": [f"{PROGRAM_FILES}\\Kaspersky Lab\\Kaspersky*\\avp.exe"],
    "avast": [f"{PROGRAM_FILES}\\AVAST Software\\Avast\\AvastUI.exe"],
    "аваст": [f"{PROGRAM_FILES}\\AVAST Software\\Avast\\AvastUI.exe"],
    "avg": [f"{PROGRAM_FILES}\\AVG\\Antivirus\\AvGui.exe"],
    "avg antivirus": [f"{PROGRAM_FILES}\\AVG\\Antivirus\\AvGui.exe"],
    "avira": [f"{PROGRAM_FILES_X86}\\Avira\\Antivirus\\Avira.Systray.exe"],
    "norton": [f"{PROGRAM_FILES}\\Norton Security\\Engine*\\uiStub.exe"],
    "eset": [f"{PROGRAM_FILES}\\ESET\\ESET Security\\egui.exe"],
    "drweb": [f"{PROGRAM_FILES}\\DrWeb\\drweb.exe"],
    "доктор веб": [f"{PROGRAM_FILES}\\DrWeb\\drweb.exe"],
    "1password": [
        f"{LOCALAPPDATA}\\1Password\\app\\8\\1Password.exe",
        "https://1password.com/",
    ],
    "lastpass": ["https://lastpass.com/"],
    "bitwarden": [
        f"{LOCALAPPDATA}\\Programs\\Bitwarden\\Bitwarden.exe",
        f"{APPDATA}\\Bitwarden\\Bitwarden.exe",
        "https://vault.bitwarden.com/",
    ],
    "битварден": [f"{LOCALAPPDATA}\\Programs\\Bitwarden\\Bitwarden.exe"],
    "keepass": [f"{PROGRAM_FILES}\\KeePass*\\KeePass.exe"],
    "dashlane": ["https://www.dashlane.com/"],
    
    # ============ 🌐 FTP И УДАЛЕННЫЙ ДОСТУП ============
    "filezilla": [
        f"{PROGRAM_FILES}\\FileZilla FTP Client\\filezilla.exe",
        f"{PROGRAM_FILES_X86}\\FileZilla FTP Client\\filezilla.exe",
    ],
    "файлзилла": [f"{PROGRAM_FILES}\\FileZilla FTP Client\\filezilla.exe"],
    "winscp": [
        f"{PROGRAM_FILES}\\WinSCP\\WinSCP.exe",
        f"{PROGRAM_FILES_X86}\\WinSCP\\WinSCP.exe",
    ],
    "teamviewer": [
        f"{PROGRAM_FILES}\\TeamViewer\\TeamViewer.exe",
        f"{PROGRAM_FILES_X86}\\TeamViewer\\TeamViewer.exe",
    ],
    "тим вьювер": [f"{PROGRAM_FILES}\\TeamViewer\\TeamViewer.exe"],
    "anydesk": [
        f"{PROGRAM_FILES}\\AnyDesk\\AnyDesk.exe",
        f"{LOCALAPPDATA}\\AnyDesk\\AnyDesk.exe",
    ],
    "анидеск": [f"{PROGRAM_FILES}\\AnyDesk\\AnyDesk.exe"],
    "rustdesk": [f"{PROGRAM_FILES}\\RustDesk\\RustDesk.exe"],
    "ammyy admin": [f"{PROGRAM_FILES_X86}\\Ammyy\\AA_v*.exe"],
    "putty": [
        f"{PROGRAM_FILES}\\PuTTY\\putty.exe",
        f"{PROGRAM_FILES_X86}\\PuTTY\\putty.exe",
    ],
    "mremoteng": [f"{PROGRAM_FILES_X86}\\mRemoteNG\\mRemoteNG.exe"],
    
    # ============ 💻 БАЗЫ ДАННЫХ ============
    "mysql workbench": [
        f"{PROGRAM_FILES}\\MySQL\\MySQL Workbench*\\MySQLWorkbench.exe",
        f"{PROGRAM_FILES_X86}\\MySQL\\MySQL Workbench*\\MySQLWorkbench.exe",
    ],
    "mysql": [f"{PROGRAM_FILES}\\MySQL\\MySQL Workbench*\\MySQLWorkbench.exe"],
    "pgadmin": [
        f"{PROGRAM_FILES}\\pgAdmin*\\runtime\\pgAdmin4.exe",
        f"{PROGRAM_FILES_X86}\\pgAdmin*\\runtime\\pgAdmin4.exe",
    ],
    "postgresql": [f"{PROGRAM_FILES}\\pgAdmin*\\runtime\\pgAdmin4.exe"],
    "postgres": [f"{PROGRAM_FILES}\\pgAdmin*\\runtime\\pgAdmin4.exe"],
    "dbeaver": [
        f"{PROGRAM_FILES}\\DBeaver\\dbeaver.exe",
        f"{LOCALAPPDATA}\\Programs\\DBeaver\\dbeaver.exe",
    ],
    "mongodb compass": [
        f"{LOCALAPPDATA}\\Programs\\MongoDB Compass\\MongoDB Compass.exe",
        f"{PROGRAM_FILES}\\MongoDB Compass\\MongoDB Compass.exe",
    ],
    "mongo": [f"{LOCALAPPDATA}\\Programs\\MongoDB Compass\\MongoDB Compass.exe"],
    "mongodb": [f"{LOCALAPPDATA}\\Programs\\MongoDB Compass\\MongoDB Compass.exe"],
    "redis": [
        f"{PROGRAM_FILES}\\Redis\\redis-server.exe",
        "https://redis.io/",
    ],
    "sqlite": [
        f"{PROGRAM_FILES}\\DB Browser for SQLite\\DB Browser for SQLite.exe",
        f"{PROGRAM_FILES_X86}\\DB Browser for SQLite\\DB Browser for SQLite.exe",
    ],
    "sqlite browser": [f"{PROGRAM_FILES}\\DB Browser for SQLite\\DB Browser for SQLite.exe"],
    "heidisql": [f"{PROGRAM_FILES}\\HeidiSQL\\heidisql.exe"],
    "datagrip": [f"{PROGRAM_FILES}\\JetBrains\\DataGrip*\\bin\\datagrip64.exe"],
    
    # ============ 🖥️ ВИРТУАЛЬНЫЕ МАШИНЫ ============
    "virtualbox": [
        f"{PROGRAM_FILES}\\Oracle\\VirtualBox\\VirtualBox.exe",
        f"{PROGRAM_FILES_X86}\\Oracle\\VirtualBox\\VirtualBox.exe",
    ],
    "virtual box": [f"{PROGRAM_FILES}\\Oracle\\VirtualBox\\VirtualBox.exe"],
    "виртуалбокс": [f"{PROGRAM_FILES}\\Oracle\\VirtualBox\\VirtualBox.exe"],
    "vmware": [
        f"{PROGRAM_FILES_X86}\\VMware\\VMware Workstation\\vmware.exe",
        f"{PROGRAM_FILES}\\VMware\\VMware Workstation\\vmware.exe",
    ],
    "vmware workstation": [f"{PROGRAM_FILES_X86}\\VMware\\VMware Workstation\\vmware.exe"],
    "vmware player": [f"{PROGRAM_FILES_X86}\\VMware\\VMware Player\\vmplayer.exe"],
    "hyper-v": ["start virtmgmt.msc"],
    "wsl": ["wsl"],
    "windows subsystem": ["wsl"],
    "docker": [
        f"{PROGRAM_FILES}\\Docker\\Docker\\Docker Desktop.exe",
        f"{LOCALAPPDATA}\\Docker\\Docker Desktop.exe",
    ],
    "docker desktop": [f"{PROGRAM_FILES}\\Docker\\Docker\\Docker Desktop.exe"],
    "докер": [f"{PROGRAM_FILES}\\Docker\\Docker\\Docker Desktop.exe"],
    "vagrant": ["vagrant"],
    
    # ============ 📱 ЭМУЛЯТОРЫ ============
    "bluestacks": [
        f"{PROGRAM_FILES}\\BlueStacks_nxt\\HD-Player.exe",
        f"{LOCALAPPDATA}\\BlueStacks_nxt\\HD-Player.exe",
    ],
    "bluestacks 5": [f"{PROGRAM_FILES}\\BlueStacks_nxt\\HD-Player.exe"],
    "блюстакс": [f"{PROGRAM_FILES}\\BlueStacks_nxt\\HD-Player.exe"],
    "nox": [
        f"{PROGRAM_FILES_X86}\\Nox\\bin\\Nox.exe",
        f"{LOCALAPPDATA}\\Nox\\bin\\Nox.exe",
    ],
    "nox player": [f"{PROGRAM_FILES_X86}\\Nox\\bin\\Nox.exe"],
    "нокс": [f"{PROGRAM_FILES_X86}\\Nox\\bin\\Nox.exe"],
    "ldplayer": [f"{PROGRAM_FILES_X86}\\LDPlayer*\\dnplayer.exe"],
    "ld player": [f"{PROGRAM_FILES_X86}\\LDPlayer*\\dnplayer.exe"],
    "mumu": [f"{PROGRAM_FILES_X86}\\Netease\\MuMu Player*\\shell\\MuMuPlayer.exe"],
    "genymotion": [f"{PROGRAM_FILES}\\Genymobile\\Genymotion\\genymotion.exe"],
    
    # ============ 🤖 НЕЙРОСЕТИ И AI ============
    "chatgpt": ["https://chat.openai.com/"],
    "gpt": ["https://chat.openai.com/"],
    "gpt-4": ["https://chat.openai.com/"],
    "gpt4": ["https://chat.openai.com/"],
    "чпт": ["https://chat.openai.com/"],
    "chat gpt": ["https://chat.openai.com/"],
    "openai": ["https://openai.com/"],
    "qwen": ["https://chat.qwen.ai/"],
    "квин": ["https://chat.qwen.ai/"],
    "квен": ["https://chat.qwen.ai/"],
    "qwen ai": ["https://chat.qwen.ai/"],
    "claude": ["https://claude.ai/"],
    "клауд": ["https://claude.ai/"],
    "claude ai": ["https://claude.ai/"],
    "anthropic": ["https://claude.ai/"],
    "gemini": ["https://gemini.google.com/"],
    "гемини": ["https://gemini.google.com/"],
    "google gemini": ["https://gemini.google.com/"],
    "bard": ["https://gemini.google.com/"],
    "бард": ["https://gemini.google.com/"],
    "perplexity": ["https://www.perplexity.ai/"],
    "перплексити": ["https://www.perplexity.ai/"],
    "copilot": ["https://copilot.microsoft.com/"],
    "копайлот": ["https://copilot.microsoft.com/"],
    "microsoft copilot": ["https://copilot.microsoft.com/"],
    "github copilot": ["https://github.com/features/copilot"],
    "yandex gpt": ["https://ya.ru/ai"],
    "яндекс gpt": ["https://ya.ru/ai"],
    "ygpt": ["https://ya.ru/ai"],
    "giga chat": ["https://giga.chat/"],
    "гига чат": ["https://giga.chat/"],
    "gigachat": ["https://giga.chat/"],
    "сбер gpt": ["https://giga.chat/"],
    "sber gpt": ["https://giga.chat/"],
    "midjourney": ["https://www.midjourney.com/"],
    "миджорни": ["https://www.midjourney.com/"],
    "mj": ["https://www.midjourney.com/"],
    "stable diffusion": [
        f"{LOCALAPPDATA}\\Programs\\Stability Matrix\\Stability Matrix.exe",
        "https://stability.ai/",
    ],
    "стейбл диффьюжн": ["https://stability.ai/"],
    "automatic1111": [
        f"{USER_HOME}\\stable-diffusion-webui\\webui-user.bat",
    ],
    "comfyui": [
        f"{USER_HOME}\\ComfyUI\\run_nvidia_gpu.bat",
    ],
    "dall-e": ["https://openai.com/dall-e-3"],
    "dalle": ["https://openai.com/dall-e-3"],
    "kandinsky": ["https://fusionbrain.ai/"],
    "кандинский": ["https://fusionbrain.ai/"],
    "fusion brain": ["https://fusionbrain.ai/"],
    "hugging face": ["https://huggingface.co/"],
    "huggingface": ["https://huggingface.co/"],
    "ollama": ["ollama"],
    "lm studio": [
        f"{LOCALAPPDATA}\\Programs\\LM Studio\\LM Studio.exe",
    ],
    "lmstudio": [f"{LOCALAPPDATA}\\Programs\\LM Studio\\LM Studio.exe"],
    
    # ============ 🏪 ИНТЕРНЕТ-МАГАЗИНЫ ============
    "ozon": ["https://www.ozon.ru/"],
    "озон": ["https://www.ozon.ru/"],
    "wildberries": ["https://www.wildberries.ru/"],
    "вайлдберриз": ["https://www.wildberries.ru/"],
    "wb": ["https://www.wildberries.ru/"],
    "яндекс маркет": ["https://market.yandex.ru/"],
    "yandex market": ["https://market.yandex.ru/"],
    "aliexpress": ["https://aliexpress.ru/"],
    "али": ["https://aliexpress.ru/"],
    "алиэкспресс": ["https://aliexpress.ru/"],
    "али экспресс": ["https://aliexpress.ru/"],
    "dns": ["https://www.dns-shop.ru/"],
    "днс": ["https://www.dns-shop.ru/"],
    "ситилинк": ["https://www.citilink.ru/"],
    "citilink": ["https://www.citilink.ru/"],
    "м видео": ["https://www.mvideo.ru/"],
    "mvideo": ["https://www.mvideo.ru/"],
    "m.video": ["https://www.mvideo.ru/"],
    "эльдорадо": ["https://www.eldorado.ru/"],
    "eldorado": ["https://www.eldorado.ru/"],
    "мегамаркет": ["https://megamarket.ru/"],
    "megamarket": ["https://megamarket.ru/"],
    "сбермегамаркет": ["https://megamarket.ru/"],
    "lamoda": ["https://www.lamoda.ru/"],
    "ламода": ["https://www.lamoda.ru/"],
    "amazon": ["https://www.amazon.com/"],
    "амазон": ["https://www.amazon.com/"],
    "ebay": ["https://www.ebay.com/"],
    "ебей": ["https://www.ebay.com/"],
    "ikea": ["https://www.ikea.com/"],
    "икея": ["https://www.ikea.com/"],
    "leroy merlin": ["https://www.leroymerlin.ru/"],
    "леруа": ["https://www.leroymerlin.ru/"],
    "obi": ["https://www.obi.ru/"],
    "оби": ["https://www.obi.ru/"],
    "hoff": ["https://hoff.ru/"],
    "хофф": ["https://hoff.ru/"],
    
    # ============ 🏛️ ГОСУДАРСТВЕННЫЕ СЕРВИСЫ ============
    "госуслуги": ["https://www.gosuslugi.ru/"],
    "gosuslugi": ["https://www.gosuslugi.ru/"],
    "мос ру": ["https://www.mos.ru/"],
    "mos.ru": ["https://www.mos.ru/"],
    "мос.ру": ["https://www.mos.ru/"],
    "налог ру": ["https://www.nalog.gov.ru/"],
    "налоговая": ["https://www.nalog.gov.ru/"],
    "фнс": ["https://www.nalog.gov.ru/"],
    "пфр": ["https://pfr.gov.ru/"],
    "сфр": ["https://sfr.gov.ru/"],
    "социальный фонд": ["https://sfr.gov.ru/"],
    "пенсионный фонд": ["https://sfr.gov.ru/"],
    "мвд": ["https://мвд.рф/"],
    "гибдд": ["https://гибдд.рф/"],
    "росреестр": ["https://rosreestr.gov.ru/"],
    "егрн": ["https://rosreestr.gov.ru/"],
    "загранпаспорт": ["https://www.gosuslugi.ru/"],
    
    # ============ 🏦 БАНКИ ============
    "сбербанк": ["https://www.sberbank.ru/"],
    "sberbank": ["https://www.sberbank.ru/"],
    "сбер": ["https://www.sberbank.ru/"],
    "сбербанк онлайн": ["https://online.sberbank.ru/"],
    "т-банк": ["https://www.tbank.ru/"],
    "тинькофф": ["https://www.tbank.ru/"],
    "tinkoff": ["https://www.tbank.ru/"],
    "альфа банк": ["https://alfabank.ru/"],
    "alfabank": ["https://alfabank.ru/"],
    "альфа": ["https://alfabank.ru/"],
    "втб": ["https://www.vtb.ru/"],
    "vtb": ["https://www.vtb.ru/"],
    "газпромбанк": ["https://www.gazprombank.ru/"],
    "открытие": ["https://www.open.ru/"],
    "райффайзен": ["https://www.raiffeisen.ru/"],
    "raiffeisen": ["https://www.raiffeisen.ru/"],
    "россельхозбанк": ["https://www.rshb.ru/"],
    "почта банк": ["https://www.pochtabank.ru/"],
    "совкомбанк": ["https://www.sovcombank.ru/"],
    "мтс банк": ["https://www.mtsbank.ru/"],
    "home credit": ["https://www.homecredit.ru/"],
    "хоум кредит": ["https://www.homecredit.ru/"],
    
    # ============ 🎓 ОБРАЗОВАНИЕ ============
    "решу огэ": ["https://oge.sdamgia.ru/"],
    "решу егэ": ["https://ege.sdamgia.ru/"],
    "sdamgia": ["https://sdamgia.ru/"],
    "сдам гиа": ["https://sdamgia.ru/"],
    "фоксфорд": ["https://foxford.ru/"],
    "foxford": ["https://foxford.ru/"],
    "университет": ["https://universarium.org/"],
    "stepik": ["https://stepik.org/"],
    "степик": ["https://stepik.org/"],
    "coursera": ["https://www.coursera.org/"],
    "курсера": ["https://www.coursera.org/"],
    "udemy": ["https://www.udemy.com/"],
    "юдеми": ["https://www.udemy.com/"],
    "edx": ["https://www.edx.org/"],
    "khan academy": ["https://www.khanacademy.org/"],
    "хан академия": ["https://www.khanacademy.org/"],
    "дуолинго": ["https://www.duolingo.com/"],
    "duolingo": ["https://www.duolingo.com/"],
    "skillbox": ["https://skillbox.ru/"],
    "скиллбокс": ["https://skillbox.ru/"],
    "нетология": ["https://netology.ru/"],
    "geekbrains": ["https://gb.ru/"],
    "гикбрейнс": ["https://gb.ru/"],
    "яндекс практикум": ["https://practicum.yandex.ru/"],
    "yandex practicum": ["https://practicum.yandex.ru/"],
    "html academy": ["https://htmlacademy.ru/"],
    "хтмл академия": ["https://htmlacademy.ru/"],
    "codecademy": ["https://www.codecademy.com/"],
    "leetcode": ["https://leetcode.com/"],
    "leetcode ru": ["https://leetcode.com/"],
    "codewars": ["https://www.codewars.com/"],
    "codeforces": ["https://codeforces.com/"],
    "хакерранк": ["https://www.hackerrank.com/"],
    "hackerrank": ["https://www.hackerrank.com/"],
    "wikipedia": ["https://ru.wikipedia.org/"],
    "википедия": ["https://ru.wikipedia.org/"],
    "wiki": ["https://ru.wikipedia.org/"],
    "вики": ["https://ru.wikipedia.org/"],
    "habr": ["https://habr.com/"],
    "хабр": ["https://habr.com/"],
    "stackoverflow": ["https://stackoverflow.com/"],
    "stack overflow": ["https://stackoverflow.com/"],
    "стэковерфлоу": ["https://stackoverflow.com/"],
    "github": ["https://github.com/"],
    "гитхаб": ["https://github.com/"],
    "gitlab": ["https://gitlab.com/"],
    "гитлаб": ["https://gitlab.com/"],
    "bitbucket": ["https://bitbucket.org/"],
    
    # ============ 📰 НОВОСТИ И МЕДИА ============
    "mail.ru": ["https://mail.ru/"],
    "майл ру": ["https://mail.ru/"],
    "яндекс почта": ["https://mail.yandex.ru/"],
    "yandex mail": ["https://mail.yandex.ru/"],
    "gmail": ["https://mail.google.com/"],
    "гмейл": ["https://mail.google.com/"],
    "google mail": ["https://mail.google.com/"],
    "outlook web": ["https://outlook.live.com/"],
    "outlook.com": ["https://outlook.live.com/"],
    "protonmail": ["https://mail.proton.me/"],
    "proton mail": ["https://mail.proton.me/"],
    "рамблер": ["https://www.rambler.ru/"],
    "rambler": ["https://www.rambler.ru/"],
    "яндекс новости": ["https://news.yandex.ru/"],
    "ria": ["https://ria.ru/"],
    "риа новости": ["https://ria.ru/"],
    "таcc": ["https://tass.ru/"],
    "tass": ["https://tass.ru/"],
    "лента ру": ["https://lenta.ru/"],
    "lenta": ["https://lenta.ru/"],
    "рбк": ["https://www.rbc.ru/"],
    "rbc": ["https://www.rbc.ru/"],
    "bbc": ["https://www.bbc.com/"],
    "cnn": ["https://www.cnn.com/"],
    "ny times": ["https://www.nytimes.com/"],
    "the times": ["https://www.thetimes.com/"],
    "reuters": ["https://www.reuters.com/"],
    "медиазона": ["https://zona.media/"],
    "медуза": ["https://meduza.io/"],
    "meduza": ["https://meduza.io/"],
    "новая газета": ["https://novayagazeta.ru/"],
    "коммерсант": ["https://www.kommersant.ru/"],
    "ведомости": ["https://www.vedomosti.ru/"],
    
    # ============ 🚗 СЕРВИСЫ ============
    "яндекс такси": ["https://taxi.yandex.ru/"],
    "такси": ["https://taxi.yandex.ru/"],
    "ситимобил": ["https://citymobil.ru/"],
    "сити мобил": ["https://citymobil.ru/"],
    "uber": ["https://www.uber.com/"],
    "убер": ["https://www.uber.com/"],
    "яндекс еда": ["https://eda.yandex.ru/"],
    "delivery club": ["https://www.delivery-club.ru/"],
    "деливери клаб": ["https://www.delivery-club.ru/"],
    "самокат": ["https://samokat.ru/"],
    "лавка": ["https://lavka.yandex.ru/"],
    "яндекс лавка": ["https://lavka.yandex.ru/"],
    "2gis": ["https://2gis.ru/"],
    "2 гис": ["https://2gis.ru/"],
    "дубль гис": ["https://2gis.ru/"],
    "google maps": ["https://maps.google.com/"],
    "гугл карты": ["https://maps.google.com/"],
    "яндекс карты": ["https://yandex.ru/maps/"],
    "яндекс навигатор": ["https://yandex.ru/maps/"],
    "aviasales": ["https://www.aviasales.ru/"],
    "авиасейлс": ["https://www.aviasales.ru/"],
    "туту": ["https://www.tutu.ru/"],
    "tutu": ["https://www.tutu.ru/"],
    "booking": ["https://www.booking.com/"],
    "букинг": ["https://www.booking.com/"],
    "airbnb": ["https://www.airbnb.com/"],
    "аирбнб": ["https://www.airbnb.com/"],
    "ostrovok": ["https://ostrovok.ru/"],
    "островок": ["https://ostrovok.ru/"],
    "cian": ["https://www.cian.ru/"],
    "циан": ["https://www.cian.ru/"],
    "авито": ["https://www.avito.ru/"],
    "avito": ["https://www.avito.ru/"],
    "юла": ["https://youla.ru/"],
    "youla": ["https://youla.ru/"],
    "headhunter": ["https://hh.ru/"],
    "hh": ["https://hh.ru/"],
    "хх ру": ["https://hh.ru/"],
    "работа ру": ["https://rabota.ru/"],
    "superjob": ["https://www.superjob.ru/"],
    "суперджоб": ["https://www.superjob.ru/"],
    
    # ============ 🎨 РАЗНОЕ ============
    "git": ["git"],
    "git bash": [f"{PROGRAM_FILES}\\Git\\bin\\bash.exe", f"{PROGRAM_FILES}\\Git\\git-bash.exe"],
    "node": ["node"],
    "nodejs": ["node"],
    "npm": ["npm"],
    "python": ["python"],
    "pip": ["pip"],
    "java": ["java"],
    "rust": ["rustc"],
    "cargo": ["cargo"],
    "go": ["go"],
    "ruby": ["ruby"],
    "php": ["php"],
    "perl": ["perl"],
    "curl": ["curl"],
    "wget": ["wget"],
}


# ================= ФУНКЦИЯ ОТКРЫТИЯ ПРИЛОЖЕНИЙ =================

def open_application(app_name: str) -> str:
    """Умный поиск приложений: словарь → where → Program Files → AppData → реестр → интернет."""
    
    app_lower = app_name.lower().strip()
    
    # ===== ШАГ 1: Проверяем словарь =====
    if app_lower in STANDARD_APPS:
        for candidate in STANDARD_APPS[app_lower]:
            try:
                # URL — открываем в браузере
                if candidate.startswith("http"):
                    webbrowser.open(candidate)
                    return f"✅ Открыт сайт: {candidate}"
                
                # Команда типа "calc", "explorer"
                if candidate in ["calc", "explorer", "mspaint", "cmd", "powershell", "taskmgr", "wsl", "git", "node", "npm", "python", "pip", "java", "rustc", "cargo", "go", "ruby", "php", "perl", "curl", "wget", "ollama", "code", "ffmpeg"]:
                    subprocess.Popen(candidate, shell=True)
                    return f"✅ Открыто: {app_name}"
                
                # Команды со "start"
                if candidate.startswith("start "):
                    subprocess.Popen(candidate, shell=True)
                    return f"✅ Открыто: {app_name}"
                
                # Путь к файлу
                expanded = os.path.expandvars(candidate)
                
                # Если есть wildcards (*), ищем файлы
                if "*" in expanded:
                    matches = glob.glob(expanded)
                    if matches:
                        subprocess.Popen([matches[0]])
                        return f"✅ Открыто: {app_name} ({os.path.basename(matches[0])})"
                    continue
                
                # Обычный путь
                if os.path.exists(expanded):
                    subprocess.Popen([expanded])
                    return f"✅ Открыто: {app_name}"
            except Exception:
                continue
    
    # ===== ШАГ 2: Ищем через where =====
    try:
        result = subprocess.run(
            f"where {app_name}", shell=True,
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0 and result.stdout.strip():
            exe_path = result.stdout.strip().split('\n')[0].strip()
            subprocess.Popen([exe_path])
            return f"✅ Найдено через where: {os.path.basename(exe_path)}"
    except Exception:
        pass
    
    # ===== ШАГ 3: Поиск в стандартных местах =====
    search_patterns = [
        f"C:\\Program Files\\*{app_lower}*\\*.exe",
        f"C:\\Program Files (x86)\\*{app_lower}*\\*.exe",
        f"{LOCALAPPDATA}\\*{app_lower}*\\*.exe",
        f"{APPDATA}\\*{app_lower}*\\*.exe",
        f"{USER_HOME}\\*{app_lower}*\\*.exe",
        f"D:\\*{app_lower}*\\*.exe",
        f"E:\\*{app_lower}*\\*.exe",
    ]
    
    for pattern in search_patterns:
        try:
            matches = glob.glob(pattern, recursive=True)
            for match in matches[:3]:  # Берём первые 3 совпадения
                if os.path.basename(match).lower().startswith(app_lower.replace(" ", "")):
                    subprocess.Popen([match])
                    return f"✅ Найдено в системе: {os.path.basename(match)}"
        except Exception:
            continue
    
    # ===== ШАГ 4: Поиск в реестре Windows =====
    try:
        import winreg
        uninstall_keys = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        ]
        
        for key_path in uninstall_keys:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                    i = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                try:
                                    name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                    if app_lower in name.lower():
                                        try:
                                            install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                            for file in os.listdir(install_location):
                                                if file.lower().endswith('.exe') and app_lower.replace(" ", "") in file.lower():
                                                    exe_path = os.path.join(install_location, file)
                                                    subprocess.Popen([exe_path])
                                                    return f"✅ Найдено в реестре: {name}"
                                        except:
                                            pass
                                except:
                                    pass
                            i += 1
                        except OSError:
                            break
            except:
                continue
    except Exception:
        pass
    
    # ===== ШАГ 5: Поиск в интернете =====
    blacklisted_domains = ["wikipedia.org", "wiki", "youtube.com/watch", "dzen.ru", "vc.ru"]
    
    try:
        results = list(DDGS().text(f"{app_name} официальный сайт", region="ru-ru", max_results=5))
        if results:
            for r in results:
                url = r['href']
                if any(bad in url.lower() for bad in blacklisted_domains):
                    continue
                webbrowser.open(url)
                return f"🌐 Открыт официальный сайт: {r['title']}"
    except Exception:
        pass
    
    return f"❌ Не удалось найти '{app_name}'. Убедитесь что приложение установлено или уточните название."


def open_site_smart(url: str) -> str:
    """Умное открытие сайта: если нет URL — ищет в интернете."""
    if url and (url.startswith("http") or "." in url):
        if not url.startswith("http"):
            url = "https://" + url
        webbrowser.open(url)
        return f"✅ Открываю {url}"
    
    try:
        results = list(DDGS().text(url + " официальный сайт", region="ru-ru", max_results=3))
        if results:
            found_url = results[0]['href']
            webbrowser.open(found_url)
            return f"🌐 Нашёл и открываю: {results[0]['title']} ({found_url})"
    except Exception as e:
        return f"Ошибка поиска: {e}"
    
    return f"❌ Не удалось найти сайт: {url}"

def web_search(query: str) -> str:
    try:
        results = list(DDGS().text(query, region="ru-ru", max_results=6))
        if not results: return "Ничего не найдено."
        return "\n\n".join(f"{i}. {r['title']}\n   {r['body']}" for i, r in enumerate(results, 1))
    except Exception as e: return f"Ошибка: {e}"

def web_news(query: str = "") -> str:
    try:
        results = list(DDGS().news(query, region="ru-ru", max_results=5))
        if not results: return "Новости не найдены."
        return "\n\n".join(f"• {r['title']}\n  {r['body'][:150]}" for r in results)
    except Exception as e: return f"Ошибка: {e}"

def scrape_web_page(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'utf-8'
        if 'raw.githubusercontent.com' in url: return response.text[:8000]
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']): tag.decompose()
        return soup.get_text(separator='\n')[:5000]
    except Exception as e: return f"Ошибка: {e}"


# ================= 4. ИНСТРУМЕНТЫ ДЛЯ АНАЛИТИКА =================

def _confirm(question: str) -> bool:
    speak(question)
    answer = listen_until_silence(max_duration=8, silence_duration=1.0)
    return any(w in answer.lower() for w in ("да", "ага", "конечно", "ок", "давай", "yes"))

def create_file(name: str, content: str, file_type: str = "txt", location: str = "work") -> str:
    if not name.endswith(f".{file_type}"): name += f".{file_type}"
    path = os.path.join(DESKTOP_DIR if location == "desktop" else WORK_DIR, os.path.basename(name))
    try:
        with open(path, "w", encoding="utf-8") as f: f.write(content)
        return f"✅ Файл создан: {path} ({len(content)} символов)"
    except Exception as e: return f"Ошибка: {e}"

def edit_file(name: str, new_content: str, location: str = "work") -> str:
    path = os.path.join(DESKTOP_DIR if location == "desktop" else WORK_DIR, os.path.basename(name))
    if not os.path.exists(path): return f"Файл не найден: {name}"
    try:
        with open(path, "w", encoding="utf-8") as f: f.write(new_content)
        return f"✅ Файл обновлён: {path} ({len(new_content)} символов)"
    except Exception as e: return f"Ошибка: {e}"

def delete_file(name: str, location: str = "work") -> str:
    path = os.path.join(DESKTOP_DIR if location == "desktop" else WORK_DIR, os.path.basename(name))
    if not os.path.exists(path): return f"Файл не найден: {name}"
    if not _confirm(f"Удалить {name}? Скажите да или нет."): return "Удаление отменено."
    try:
        os.remove(path)
        return f"✅ Файл удалён: {name}"
    except Exception as e: return f"Ошибка: {e}"

def move_file(old_name: str, new_name: str, location: str = "work") -> str:
    folder = DESKTOP_DIR if location == "desktop" else WORK_DIR
    old_path, new_path = os.path.join(folder, os.path.basename(old_name)), os.path.join(folder, os.path.basename(new_name))
    if not os.path.exists(old_path): return f"Файл не найден: {old_name}"
    try:
        os.rename(old_path, new_path)
        return f"✅ Файл переименован: {old_name} → {new_name}"
    except Exception as e: return f"Ошибка: {e}"

def run_command(cmd: str) -> str:
    cmd_stripped = cmd.strip().lower()
    is_safe = any(cmd_stripped.startswith(s) for s in SAFE_COMMANDS)
    if not is_safe and not _confirm(f"Выполнить {cmd}? Скажите да или нет."): return "Отклонено."
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip() or "Выполнено."
    except Exception as e: return f"Ошибка: {e}"

def read_file_any(name: str, location: str = "work") -> str:
    path = os.path.join(DESKTOP_DIR if location == "desktop" else WORK_DIR, os.path.basename(name))
    if not os.path.exists(path): return f"Файл не найден: {name}"
    try:
        with open(path, "r", encoding="utf-8") as f: return f.read()[:8000]
    except Exception as e: return f"Ошибка: {e}"

def list_all_files(location: str = "work") -> str:
    folder = DESKTOP_DIR if location == "desktop" else WORK_DIR
    files = os.listdir(folder)
    if not files: return "Папка пуста."
    return f"Файлы в {folder}:\n" + "\n".join(f"  • {f}" for f in files)

def download_file(url: str, name: str, file_type: str = "any", location: str = "work") -> str:
    if file_type != "any" and not name.endswith(f".{file_type}"): name += f".{file_type}"
    path = os.path.join(DESKTOP_DIR if location == "desktop" else WORK_DIR, os.path.basename(name))
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        with open(path, 'wb') as f: f.write(response.content)
        return f"Скачан: {path} ({len(response.content)} байт)"
    except Exception as e: return f"Ошибка: {e}"

def search_github_code(query: str) -> str:
    try:
        results = list(DDGS().text(f"site:github.com {query}", region="ru-ru", max_results=5))
        return "\n\n".join(f"{r['title']}\n  {r['href']}" for r in results) if results else "Не найдено"
    except Exception as e: return f"Ошибка: {e}"


# ================= 5. ИНСТРУМЕНТЫ ДЛЯ АРХИТЕКТОРА =================

def create_python_file(name: str, code: str, location: str = "work") -> str:
    if not code or len(code.strip()) < 50: return "ОШИБКА: код слишком короткий"
    path = os.path.join(DESKTOP_DIR if location == "desktop" else WORK_DIR, os.path.basename(name))
    if not path.endswith(".py"): path += ".py"
    try:
        with open(path, "w", encoding="utf-8") as f: f.write(code)
        return f"✅ Файл создан: {path} ({len(code)} символов, {code.count(chr(10))+1} строк)"
    except Exception as e: return f"Ошибка: {e}"

def run_python_file(name: str, location: str = "work") -> str:
    path = os.path.join(DESKTOP_DIR if location == "desktop" else WORK_DIR, os.path.basename(name))
    if not path.endswith(".py"): path += ".py"
    if not os.path.exists(path): return f"Файл не найден: {os.path.basename(path)}"
    if not _confirm(f"Запустить {os.path.basename(path)}? Да или нет?"): return "Отменено."
    try:
        if os.name == 'nt':
            subprocess.Popen(["python", path], creationflags=subprocess.CREATE_NEW_CONSOLE)
            return f"Запускаю {os.path.basename(path)}"
        else:
            out = subprocess.run(["python", path], capture_output=True, text=True, timeout=60)
            return out.stdout or "Выполнено."
    except Exception as e: return f"Ошибка: {e}"

def write_and_run_python(code: str, name: str = "temp.py", location: str = "work") -> str:
    created = create_python_file(name, code, location)
    if "Ошибка" in created: return created
    filename = os.path.basename(created.split(": ")[-1]).split(" ")[0]
    return run_python_file(filename, location)

def install_package(package_name: str) -> str:
    return run_command(f"pip install {package_name}")


# ================= 6. НАБОРЫ ИНСТРУМЕНТОВ =================

ALICE_TOOLS = [
    {"type": "function", "function": {"name": "analyze_image", "description": "Проанализировать изображение по пути к файлу.", "parameters": {"type": "object", "properties": {"image_path": {"type": "string"}, "question": {"type": "string"}}, "required": ["image_path"]}}},
    {"type": "function", "function": {"name": "capture_and_analyze_camera", "description": "Сделать снимок с веб-камеры и проанализировать его.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "take_screenshot_and_analyze", "description": "Сделать скриншот ЭКРАНА КОМПЬЮТЕРА и проанализировать его.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "list_desktop_files", "description": "Показать СПИСОК файлов и папок на верхнем уровне рабочего стола.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "read_document", "description": "Прочитать PDF или изображение документа и извлечь текст.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "read_table", "description": "Прочитать таблицу (CSV, Excel) и вернуть содержимое.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "deep_research", "description": "Глубокое исследование: обходит 15-50 сайтов. ОБЯЗАТЕЛЬНО используй для любых вопросов.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "depth": {"type": "string", "enum": ["fast", "medium", "deep"]}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "open_application", "description": "Открыть приложение или сайт по имени (калькулятор, телеграм, youtube, решу огэ, dns, steam, visual studio и т.д.).", "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]}}},
    {"type": "function", "function": {"name": "open_site_smart", "description": "Открыть сайт по URL или по названию (найдёт в интернете).", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Быстрый поиск в интернете.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_news", "description": "Последние новости.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "scrape_web_page", "description": "Прочитать содержимое конкретного URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]

ANALYST_TOOLS = ALICE_TOOLS + [
    {"type": "function", "function": {"name": "create_file", "description": "Создать файл любого типа.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "content": {"type": "string"}, "file_type": {"type": "string", "default": "txt"}, "location": {"type": "string", "enum": ["work", "desktop"], "default": "work"}}, "required": ["name", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Редактировать существующий файл.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "new_content": {"type": "string"}, "location": {"type": "string", "enum": ["work", "desktop"], "default": "work"}}, "required": ["name", "new_content"]}}},
    {"type": "function", "function": {"name": "delete_file", "description": "Удалить файл.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "location": {"type": "string", "enum": ["work", "desktop"], "default": "work"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "move_file", "description": "Переименовать или переместить файл.", "parameters": {"type": "object", "properties": {"old_name": {"type": "string"}, "new_name": {"type": "string"}, "location": {"type": "string", "enum": ["work", "desktop"], "default": "work"}}, "required": ["old_name", "new_name"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Выполнить системную команду.", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
    {"type": "function", "function": {"name": "read_file_any", "description": "Прочитать любой текстовый файл.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "location": {"type": "string", "enum": ["work", "desktop"], "default": "work"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "list_all_files", "description": "Показать все файлы в папке.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "enum": ["work", "desktop"], "default": "work"}}}}},
    {"type": "function", "function": {"name": "download_file", "description": "Скачать файл по URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "name": {"type": "string"}, "file_type": {"type": "string", "default": "any"}, "location": {"type": "string", "enum": ["work", "desktop"], "default": "work"}}, "required": ["url", "name"]}}},
    {"type": "function", "function": {"name": "search_github_code", "description": "Поиск готового кода на GitHub.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

ARCHITECT_TOOLS = ANALYST_TOOLS + [
    {"type": "function", "function": {"name": "create_python_file", "description": "Создать питон-файл с большим качественным кодом (200-500 строк).", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "code": {"type": "string"}, "location": {"type": "string", "enum": ["work", "desktop"], "default": "work"}}, "required": ["name", "code"]}}},
    {"type": "function", "function": {"name": "run_python_file", "description": "Запустить питон-файл.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "location": {"type": "string", "enum": ["work", "desktop"], "default": "work"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "write_and_run_python", "description": "Создать и сразу запустить питон-код.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}, "name": {"type": "string", "default": "temp.py"}, "location": {"type": "string", "enum": ["work", "desktop"], "default": "work"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "install_package", "description": "Установить Python пакет через pip.", "parameters": {"type": "object", "properties": {"package_name": {"type": "string"}}, "required": ["package_name"]}}},
]

ALICE_FUNCTIONS = {"analyze_image": analyze_image, "capture_and_analyze_camera": capture_and_analyze_camera, "take_screenshot_and_analyze": take_screenshot_and_analyze, "list_desktop_files": list_desktop_files, "read_document": read_document, "read_table": read_table, "deep_research": deep_research, "open_application": open_application, "open_site_smart": open_site_smart, "web_search": web_search, "web_news": web_news, "scrape_web_page": scrape_web_page}
ANALYST_FUNCTIONS = {**ALICE_FUNCTIONS, "create_file": create_file, "edit_file": edit_file, "delete_file": delete_file, "move_file": move_file, "run_command": run_command, "read_file_any": read_file_any, "list_all_files": list_all_files, "download_file": download_file, "search_github_code": search_github_code}
ARCHITECT_FUNCTIONS = {**ANALYST_FUNCTIONS, "create_python_file": create_python_file, "run_python_file": run_python_file, "write_and_run_python": write_and_run_python, "install_package": install_package}


# ================= 7. СИСТЕМНЫЕ ПРОМПТЫ =================

SYSTEM_ALICE = (
    "Ты — дружелюбная голосовая помощница Алиса. Отвечай на русском.\n\n"
    
    "🔴 КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:\n"
    "1. ВСЕГДА вызывай инструменты через function calling — НИКОГДА не притворяйся что уже сделала\n"
    "2. НИКОГДА не вызывай один и тот же инструмент с теми же аргументами дважды\n"
    "3. Если не знаешь URL сайта — используй open_application (он сам найдёт) или сначала web_search\n"
    "4. После выполнения инструмента давай КОРОТКИЙ ответ (1-2 предложения)\n\n"
    
    "ТВОИ ВОЗМОЖНОСТИ:\n"
    "• analyze_image, capture_and_analyze_camera, take_screenshot_and_analyze — зрение\n"
    "• list_desktop_files — список файлов на рабочем столе\n"
    "• read_document, read_table — чтение документов и таблиц\n"
    "• deep_research — исследование 15-50 сайтов\n"
    "• open_application — открывать приложения И САЙТЫ по имени (500+ программ и сайтов!)\n"
    "• open_site_smart — открывать сайт по URL или названию\n"
    "• web_search, web_news — поиск и новости\n"
    "• scrape_web_page — прочитать URL\n\n"
    
    "🔴 СТРОГИЕ ОГРАНИЧЕНИЯ:\n"
    "❌ НЕ СОЗДАВАЙ и НЕ РЕДАКТИРУЙ файлы\n"
    "❌ НЕ ПИШИ код — передавай Архитектору\n"
    "❌ НЕ ПИШИ 'открываю' или 'отправлю ссылку' БЕЗ реального вызова инструмента\n\n"
    
    "ПРИМЕРЫ ПРАВИЛЬНЫХ ДЕЙСТВИЙ:\n"
    "• 'Открой ютуб' → open_application('youtube') → 'Открываю YouTube!'\n"
    "• 'Открой решу огэ' → open_application('решу огэ') → 'Открываю сайт для подготовки к ОГЭ'\n"
    "• 'Открой стим' → open_application('steam') → 'Открываю Steam!'\n"
    "• 'Открой визуал студио' → open_application('visual studio') → 'Открываю Visual Studio!'\n"
    "• 'Открой телеграм' → open_application('telegram') → 'Открываю Telegram!'\n"
    "• 'Что на экране' → take_screenshot_and_analyze → подробное описание\n"
    "• 'Расскажи про чёрные дыры' → deep_research → развёрнутый ответ\n\n"
    
    "ПРАВИЛА ОТВЕТА:\n"
    "1. Для информационного вопроса — deep_research\n"
    "2. Для открытия сайта/приложения — open_application\n"
    "3. После инструмента — короткий текстовый ответ\n"
    "4. НЕ возвращай JSON\n"
)

SYSTEM_ANALYST = (
    "Ты — умный аналитик с логическим мышлением. Отвечай на русском.\n\n"
    "ТВОИ ВОЗМОЖНОСТИ:\n"
    "• Всё что может Алиса\n"
    "• create_file, edit_file, delete_file, move_file — управление файлами\n"
    "• run_command — выполнять системные команды\n"
    "• read_file_any, list_all_files — читать и просматривать содержимое папок\n"
    "• download_file, search_github_code — скачивание и поиск\n\n"
    "ПРАВИЛА:\n"
    "1. Для сложных вопросов используй deep_research\n"
    "2. Для создания файлов — create_file\n"
    "3. Отвечай КРАТКО после инструментов\n"
    "4. НЕ возвращай JSON\n"
    "5. Если просят написать программу — скажи: 'Это задача для Архитектора.'\n"
    "6. НЕ вызывай один и тот же инструмент дважды\n\n"
    f"Рабочая папка: {WORK_DIR}\nРабочий стол: {DESKTOP_DIR}"
)

SYSTEM_ARCHITECT = (
    "Ты — эксперт-архитектор программного обеспечения. Пишешь качественный код 200-500 строк. Отвечай на русском.\n\n"
    "ТВОИ ВОЗМОЖНОСТИ:\n"
    "• Всё что может аналитик\n"
    "• create_python_file, run_python_file, write_and_run_python, install_package\n\n"
    "СТРОГИЙ АЛГОРИТМ:\n"
    "1. search_github_code — поиск примеров\n"
    "2. scrape_web_page — изучить подходы\n"
    "3. install_package — установить библиотеки\n"
    "4. create_python_file — создать большой качественный файл\n"
    "5. run_python_file — проверить\n\n"
    "ТРЕБОВАНИЯ К КОДУ: GUI, обработка ошибок, комментарии на русском, горячие клавиши, 200-500 строк минимум.\n"
    "ПРАВИЛА: НЕ возвращай JSON. НЕ пиши код в тексте — только через инструменты. НЕ вызывай инструменты повторно.\n\n"
    f"Рабочая папка: {WORK_DIR}\nРабочий стол: {DESKTOP_DIR}"
)


# ================= 8. МАРШРУТИЗАЦИЯ =================

history_alice = [{"role": "system", "content": SYSTEM_ALICE}]
history_analyst = [{"role": "system", "content": SYSTEM_ANALYST}]
history_architect = [{"role": "system", "content": SYSTEM_ARCHITECT}]

def _trim_history(history, max_chars):
    system = history[0]
    rest = history[1:]
    for m in rest:
        if len(str(m.get("content", ""))) > 4000: m["content"] = str(m["content"])[:4000] + "..."
    total = sum(len(str(m.get("content", ""))) for m in rest)
    while total > max_chars and len(rest) > 2:
        removed = rest.pop(0)
        total -= len(str(removed.get("content", "")))
    history[:] = [system] + rest

def classify_request(text: str) -> str:
    text_lower = text.lower()
    
    transfer_markers = ["запусти аналитик", "вызови аналитика", "пусть аналитик", "передай аналитику", "аналитик сделай", "архитектор сделай", "запусти архитектора", "вызови программиста", "программист напиши", "передай программисту"]
    if any(m in text_lower for m in transfer_markers):
        return "architect" if "архитектор" in text_lower or "программист" in text_lower else "analyst"
    
    architect_markers = ["напиши код", "написать код", "можешь написать код", "напиши программу", "написать программу", "создай программу", "сделай программу", "напиши скрипт", "написать скрипт", "создай скрипт", "напиши автокликер", "создай автокликер", "напиши парсер", "создай парсер", "напиши бота", "создай бота", "напиши игру", "создай игру", "напиши приложение", "создай приложение", "напиши калькулятор", "создай калькулятор", "напиши на питоне", "написать на питоне", "программа на python", "код на питоне", "сделай скрипт", "разработай программу", "напиши gui", "создай gui", "напиши софт", "напиши утилиту", "напиши инструмент"]
    if any(m in text_lower for m in architect_markers): return "architect"
    
    analyst_markers = ["создай файл", "открой файл", "удали файл", "переименуй файл", "измени файл", "отредактируй файл", "сохрани в файл", "запиши в файл", "выполни команду", "установи библиотеку", "pip install", "скачай файл", "скачать документ", "прочитай файл", "покажи файлы", "какие файлы есть", "список файлов", "создай txt", "создай csv", "создай json", "создай markdown", "создай конфиг", "обнови настройки", "там будет написано", "там будет записано", "будет написано", "посмотри что в папке", "загляни в папку", "что внутри папки", "содержимое папки", "посмотри внутри"]
    if any(m in text_lower for m in analyst_markers): return "analyst"
    
    return "alice"

def _extract_tool_calls_from_text(text: str):
    if not text: return []
    found = []
    for candidate in re.findall(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text) + re.findall(r'\{[^{}]*?"name"[^{}]*?"arguments"[^{}]*?\}', text):
        try:
            data = json.loads(candidate)
            if "name" in data and ("arguments" in data or "parameters" in data):
                found.append((data["name"], data.get("arguments") or data.get("parameters") or {}))
        except: continue
    return found

def _clean_response(text: str) -> str:
    if not text: return ""
    text = re.sub(r'```(?:json)?\s*\{[\s\S]*?\}\s*```', '', text)
    text = re.sub(r'\{[^{}]*?"name"[^{}]*?"arguments"[^{}]*?\}', '', text)
    if text.strip().startswith('{') and text.strip().endswith('}'):
        try:
            if "name" in json.loads(text.strip()): return ""
        except: pass
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`\n]+`', '', text)
    lines = [l for l in text.split('\n') if not l.strip().startswith(('import ', 'from ', 'def ', 'class ', 'if __', 'return ', 'print('))]
    text = re.sub(r'https?://\S+', '', '\n'.join(lines))
    return re.sub(r'\s+', ' ', text).strip()

def _execute_tool(name: str, arguments, functions_dict) -> str:
    if name not in functions_dict: return f"Неизвестный инструмент: {name}"
    if isinstance(arguments, str):
        try: arguments = json.loads(arguments)
        except: arguments = {}
    if not isinstance(arguments, dict): arguments = {}
    try: return str(functions_dict[name](**arguments))
    except Exception as e: return f"Ошибка: {e}"

def think(user_text: str) -> str:
    level = classify_request(user_text)
    if level == "alice" and any(w in user_text.lower() for w in ["аналитик", "архитектор", "программист"]):
        level = "architect" if "архитектор" in user_text.lower() or "программист" in user_text.lower() else "analyst"
        print(f"🔄 Принудительная передача → {level}")
    
    if level == "architect": model, ctx, history, tools, funcs = ARCHITECT_MODEL, ARCHITECT_CONTEXT, history_architect, ARCHITECT_TOOLS, ARCHITECT_FUNCTIONS; icon = "💎 Архитектор"
    elif level == "analyst": model, ctx, history, tools, funcs = ANALYST_MODEL, ANALYST_CONTEXT, history_analyst, ANALYST_TOOLS, ANALYST_FUNCTIONS; icon = "🧠 Аналитик"
    else: model, ctx, history, tools, funcs = ALICE_MODEL, ALICE_CONTEXT, history_alice, ALICE_TOOLS, ALICE_FUNCTIONS; icon = "⚡ Алиса"
    
    print(f"{icon} → {model}")
    history.append({"role": "user", "content": user_text})
    _trim_history(history, ctx * 3)

    max_rounds, rounds, final_content = 5, 0, ""
    executed_calls = set()
    
    try:
        while rounds < max_rounds:
            start_time = time.time()
            resp = ollama.chat(model=model, messages=history, tools=tools, options={"num_ctx": ctx, "temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1})
            print(f"   ⏱ Ответ модели получен за {time.time() - start_time:.1f} сек")
            
            msg = resp["message"] if isinstance(resp, dict) else resp.message
            raw_content = msg.get("content", "") if isinstance(msg, dict) else (msg.content if hasattr(msg, 'content') else "")
            native_calls = msg.get("tool_calls") if isinstance(msg, dict) else (msg.tool_calls if hasattr(msg, 'tool_calls') else None) or []
            text_calls = _extract_tool_calls_from_text(raw_content)
            
            all_calls = []
            for call in native_calls:
                func = call.get("function") if isinstance(call, dict) else (call.function if hasattr(call, 'function') else None)
                if func:
                    name = func.get("name") if isinstance(func, dict) else (func.name if hasattr(func, 'name') else None)
                    args = func.get("arguments") if isinstance(func, dict) else (func.arguments if hasattr(func, 'arguments') else None)
                    if name: all_calls.append((name, args))
            for name, args in text_calls:
                if not any(n == name for n, _ in all_calls): all_calls.append((name, args))
            
            unique_calls = []
            for name, args in all_calls:
                try:
                    args_str = json.dumps(args, sort_keys=True, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                except:
                    args_str = str(args)
                call_signature = f"{name}:{args_str}"
                if call_signature not in executed_calls:
                    unique_calls.append((name, args))
                    executed_calls.add(call_signature)
                else:
                    print(f"   🚫 Дублирующий вызов пропущен: {name}")
            
            if not unique_calls:
                final_content = raw_content
                break
            
            results = []
            for name, args in unique_calls:
                print(f"   ⚙️ Вызов: {name}({args})")
                try: result = _execute_tool(name, args, funcs)
                except Exception as e: result = f"Ошибка: {e}"
                results.append(result)
                print(f"   ✓ Результат: {result[:100]}...")
            
            if native_calls:
                history.append(msg if isinstance(msg, dict) else {"role": "assistant", "content": raw_content, "tool_calls": native_calls})
                for r in results: history.append({"role": "tool", "content": r})
            else:
                tool_results = "\n\n".join(f"Инструмент {n}:\n{r}" for (n, _), r in zip(unique_calls, results))
                history.append({"role": "assistant", "content": raw_content})
                history.append({"role": "user", "content": tool_results + "\n\nДай пользователю короткий ТЕКСТОВЫЙ ответ о результате. НЕ вызывай инструменты повторно — задача уже выполнена."})
            
            rounds += 1
            _trim_history(history, ctx * 3)
        
        clean = _clean_response(final_content)
        history.append({"role": "assistant", "content": clean if clean and len(clean) >= 3 else "Готово!"})
        return history[-1]["content"]
    except KeyboardInterrupt: raise
    except Exception as e:
        error_msg = f"Ошибка: {str(e)[:100]}"
        print(error_msg)
        history.append({"role": "assistant", "content": error_msg})
        return error_msg


# ================= 9. ГЛАВНЫЙ ЦИКЛ =================
if __name__ == "__main__":
    print("=" * 60)
    print("🏠 ДОМАШНИЙ ИИ — 3 УРОВНЯ МОДЕЛЕЙ")
    print(f"   ⚡ Алиса (мультимодальная):     {ALICE_MODEL}")
    print(f"   🧠 Аналитик (логика+файлы):    {ANALYST_MODEL}")
    print(f"   💎 Архитектор (кодинг):        {ARCHITECT_MODEL}")
    print(f"   📁 Рабочая папка: {WORK_DIR}")
    print(f"   🖥 Рабочий стол: {DESKTOP_DIR}")
    print(f"   📦 В словаре: {len(STANDARD_APPS)} приложений/сайтов")
    print("   Скажите 'стоп' для выхода.")
    print("=" * 60)

    speak("Привет! Я трёхуровневый ассистент. Могу смотреть через камеру, снимать скриншоты, читать документы, анализировать таблицы, искать информацию в интернете, управлять файлами и писать программы.")

    while True:
        try:
            text = listen_until_silence()
            if not text: continue
            if any(w in text.lower() for w in ("стоп", "выход", "пока", "stop")):
                speak("До свидания!")
                break
            speak(think(text))
        except KeyboardInterrupt: break
        except Exception as e:
            print(f"Ошибка: {e}")
            traceback.print_exc()
            try: speak(f"Ошибка: {str(e)[:100]}")
            except: pass
