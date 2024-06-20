# 🎮 Counter-Strike AI Assistant Chat-Bot 🤖

Welcome to the Counter-Strike AI Assistant Chat-Bot! This nifty script brings the magic of AI-driven chat assistance to Counter-Strike 2 and other GoldSource, Source/2 games. Chat with an AI buddy or let others on the same server have fun using the power of OpenAI's GPT model or Gemini's generative AI.

**Please use this tool ethically and responsibly.**

## 🛠️ Requirements

Before diving in, make sure you have the following:

- 64-bit Windows
- Python 3.11+
- You'll need an API key to use the chat features.

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
- **Automated API Interactions**: Seamlessly connect with OpenAI's Chat Completion, Gemini's Generative Model APIs, and OpenRouter's modelbase, including access to free models, ensuring smooth and efficient data exchanges.
- **Configurable Settings**: Customize prompts and API parameters via `config.ini` or the GUI.
- **Themes**: Choose between light and dark UI themes to match your style.

Get ready to enhance your Counter-Strike experience with a chat-bot that's as smart as it is fun! 🎉

## Screenshot

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/KristjanPikhof/CS2-Chatbot-with-GPT-Gemini-OpenRouter-integrations/assets/60576985/6efc09be-d1cf-4e81-b050-4acfb8bce363" alt="Dark theme"/>
    <img src="https://github.com/KristjanPikhof/CS2-Chatbot-with-GPT-Gemini-OpenRouter-integrations/assets/60576985/fdeb11b5-5016-4897-9042-dea275c41686" alt="Light theme"/>
</div>


