# Moya AI - Personal Voice Assistant


**Why Moya?**  
The name "Moya" comes from my personal touch—short, friendly, and easy to say as a wake word for your personal AI assistant.  

Moya AI is an **English voice assistant** built with **Python**, **Whisper ASR**, and **ElevenLabs/pyttsx3 TTS** that can respond to almost any PC or online command—fast, smooth, and intelligent.  

---

## Features

- **Wake-word detection:** Say "**Moya**" to activate for low-volume speech.  
- **Advanced speech recognition:** Powered by Whisper medium model for accurate English transcription.  
- **Text-to-Speech (TTS):** Female voice using ElevenLabs Bella (premium) or pyttsx3 fallback.  
- **PC Control:** Open/close YouTube, manage Chrome/Edge windows.  
- **Online Search & Media:** Google search, play music on YouTube, YouTube Music search.  
- **Low-volume support:** Detects soft speech and prompts politely if wake-word is needed.  
- **Threaded & responsive:** Handles multiple tasks without freezing the assistant.  
- **Extensible:** Easy to add more PC commands or API integrations.

---

## 🔗 API System

Moya AI integrates multiple APIs for enhanced functionality:  

- **ElevenLabs TTS API:** For natural, high-quality female voice. Requires `ELEVENLABS_API_KEY` as environment variable.  
- **Whisper ASR:** Offline transcription with support for accurate English speech recognition.  
- **YouTube & Google automation:** Uses `pywhatkit` and webbrowser for searches and media playback.

---

## Prerequisites & Setup

- **Python 3.10+**  
- Install required packages:

```bash
pip install -r requirements.txt
