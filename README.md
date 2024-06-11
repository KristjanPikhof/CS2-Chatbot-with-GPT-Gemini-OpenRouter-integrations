# 🎮 Counter-Strike AI Assistant Chat-Bot 🤖

Welcome to the Counter-Strike AI Assistant Chat-Bot! This nifty script brings the magic of AI-driven chat assistance to Counter-Strike 2 and other GoldSource, Source/2 games. Chat with an AI buddy or let others on the same server have fun using the power of OpenAI's GPT model or Gemini's generative AI.

**Please use this tool ethically and responsibly.**

## 🛠️ Requirements

Before diving in, make sure you have the following:

- 64-bit Windows
- Python 3.11+ (but less than 3.12)
- `openai 0.28` (Install with `pip install -r requirements.txt`)

You'll need an OpenAI API key to use the chat features. If you're opting for Gemini's generative AI services, a Gemini API key is also required.

## 🚀 Setup

To get things rolling, enable console logging in your game:

1. **Counter-Strike Source (CS:S)**: Open the in-game developer console and type: `con_logfile <filename>; con_timestamp 1`. Do this every time you start the game.
2. **Counter-Strike Global Offensive (CS2) or Half-Life (HL)**: Add `-condebug` to your Steam game launch options.

Your console log file might be found here: `C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\console.log`.

Next, configure your bot:

- Edit `config.ini` to set `gameconlogpath`, your in-game username, and your AI API keys.
- Alternatively, use the GUI to set up everything and save the changes.

Launch the bot through the command line:

```
python chatbot.py
```

## 🤖 How It Works

This script builds upon Isaac Duarte's framework and enhances xsync3d's version with a superior user experience and a wealth of options. It keeps an eye on the game's console log for new entries, identifies blacklisted users, and forwards detected messages to OpenAI's GPT model or Gemini's generative AI. The AI-generated responses are then fed back into the game via simulated keystrokes.

For optimal performance, it's recommended to start the game before running the script. If the game isn't started beforehand, the default game will be set to CS2.

## ✨ Additional Features

The AI chat-bot is packed with cool features:
- **Hotkeys**: Control the chat-bot with keyboard shortcuts to start/stop and toggle between ALL and TEAM chats.
- **Automated API Interactions**: Seamlessly connect with OpenAI's Chat Completion and Gemini's Generative Model APIs.
- **Configurable Settings**: Customize prompts and API parameters via `config.ini` or the GUI.
- **Themes**: Choose between light and dark UI themes to match your style.

Get ready to enhance your Counter-Strike experience with a chat-bot that's as smart as it is fun! 🎉

## Screenshot
![image](https://github.com/KristjanPikhof/CS2-Chatbot-with-GPT-Gemini-integration/assets/60576985/0f477ffd-9dff-4c42-bde0-13b8edad0da4)
