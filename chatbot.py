from openai import OpenAI
import time
import random
import conparser as cp
import keyboard
import configparser
import os
import sys
import logging
import google.generativeai as genai
import utils as util

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QRadioButton,
    QButtonGroup, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s: %(message)s',
                    datefmt='%d.%m.%Y %H:%M:%S')

# Initialize the config parser
config = configparser.ConfigParser()
CONFIG_FILE = 'config.ini'

LIGHT_MODE_STYLESHEET = """
    QWidget {
        background-color: #f5f5f5;
        color: #212121;
    }
    QPushButton {
        background-color: #e0e0e0;
        color: #212121;
        border: none;
        padding: 10px 20px;
        text-transform: uppercase;
        font-weight: bold;
        border-radius: 4px; 
    }
    QPushButton:hover {
        background-color: #bdbdbd;
    }
    QPushButton:pressed {
        background-color: #9e9e9e;
    }
    QLineEdit {
        background-color: #ffffff;
        border: 1px solid #bdbdbd;
        border-radius: 4px;
        padding: 8px; 
    }

    QToolTip { 
        background-color: #ffffff; 
        color: #212121;
        border: 1px solid #bdbdbd;
        padding: 5px;
        border-radius: 4px;
    }
"""

DARK_MODE_STYLESHEET = """
    QWidget {
        background-color: #121212;
        color: #ffffff; 
    }
    QPushButton {
        background-color: #212121;
        color: white;
        border: none;
        padding: 10px 20px;
        text-transform: uppercase;
        font-weight: bold;
        border-radius: 4px; 
    }
    QPushButton:hover {
        background-color: #303030;
    }
    QPushButton:pressed {
        background-color: #424242;
    }
    QLineEdit {
        background-color: #1e1e1e;
        color: white; 
        border: 1px solid #424242;
        border-radius: 4px;
        padding: 8px;
    }

    QToolTip {
        background-color: #303030;
        color: #ffffff;
        border: 1px solid #424242;
        padding: 5px;
        border-radius: 4px;
    }
"""

# Gemini configuration
gemini_generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
gemini_safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Load settings from the config file with defaults
def get_blacklisted_usernames():
    return [username.strip() for username in
            config['SETTINGS'].get('username', '').replace('[DEAD]', '').split(';')]

def resource_path(relative_path):
    """Get the correct path to a resource file, even when running as executable."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def load_config():
    defaults = {
        'username': '',
        'chatkey': 'y',
        'gameconlogpath': '',
        'teamchatkey': 'u',
        'startstopkey': 'F10',
        'togglechatkey': 'F11',
        'openaiapikey': '',
        'geminiapikey': '',
        'openrouterapikey': '',
        'openroutermodel': '',
        'allsystemprompt': 'You are a helpful assistant in CS2',
        'teamsystemprompt': 'You are a helpful assistant in CS2',
        'togglemodelkey': 'F12'
    }
    try:
        config_path = resource_path(CONFIG_FILE) 
        print("Looking for config.ini at:", config_path)

        if not os.path.exists(config_path):
            logging.error(f"config.ini not found at: {config_path}")
        else:
            logging.info(f"config.ini found at: {config_path}")

        config.read(config_path, encoding='utf-8')
        logging.info("Configuration loaded:")
        for section in config.sections():
            for key in defaults:
                if not config[section].get(key):
                    config[section][key] = defaults[key]
                    logging.warning(
                        f"Setting '{key}' in 'config.ini' is empty. Using default: '{defaults[key]}'")
                elif key in ("openaiapikey", "geminiapikey", "openrouterapikey"):
                    logging.info(f"{key} = {'*' * len(config[section][key])}")  # Mask API keys
                else:
                    logging.info(f"{key} = {config[section][key]}")

    except Exception as e:
        logging.error(f"Error reading config file: {e}")


# Load the initial configuration
load_config()
BLACKLISTED_USERNAMES = get_blacklisted_usernames()
CON_LOG_FILE_PATH = config['SETTINGS']['gameconlogpath']
CHAT_KEY = config['SETTINGS']['chatkey']
TEAM_CHAT_KEY = config['SETTINGS']['teamchatkey']
START_STOP_KEY = config['SETTINGS']['startstopkey']
TOGGLE_CHAT_KEY = config['SETTINGS']['togglechatkey']
TOGGLE_MODEL_KEY = config['SETTINGS']['togglemodelkey']
ALL_CHAT_SYSTEM_PROMPT = config['SETTINGS']['allsystemprompt']
TEAM_CHAT_SYSTEM_PROMPT = config['SETTINGS']['teamsystemprompt']

# Fetch the AI API keys and models from the config file
OPENAI_KEY = config['SETTINGS']['openaiapikey']
GENAI_KEY = config['SETTINGS']['geminiapikey']
OPENROUTER_KEY = config['SETTINGS']['openrouterapikey']
OPENAI_MODEL = config['SETTINGS'].get('openaimodel', 'gpt-4o-mini')
OPENROUTER_MODEL = config['SETTINGS'].get('openroutermodel', 'meta-llama/llama-3.1-8b-instruct:free')
GEMINI_MODEL = config['SETTINGS'].get('geminimodel', 'gemini-1.5-flash-latest')
OPENAI_BASE_URL = "https://api.openai.com/v1/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/"

# Class to manage the running status, chat mode, and chosen AI
class Status:
    running = False
    chat_mode = "all"
    ai_model = "openai"

def _remove_hotkeys_if_exists(hotkeys):
    for hotkey in hotkeys:
        try:
            keyboard.remove_hotkey(hotkey)
        except KeyError:
            pass

def register_hotkeys():
    _remove_hotkeys_if_exists([START_STOP_KEY, TOGGLE_CHAT_KEY, TOGGLE_MODEL_KEY])
    keyboard.add_hotkey(START_STOP_KEY, toggle_status)
    keyboard.add_hotkey(TOGGLE_CHAT_KEY, toggle_chat_mode)
    keyboard.add_hotkey(TOGGLE_MODEL_KEY, toggle_model)
    logging.info(
        f"Hotkeys registered: {START_STOP_KEY} for start/stop, {TOGGLE_CHAT_KEY} for toggle chat mode")

# Function to toggle the running status
def toggle_status():
    if not Status.running:
        game = cp.detect_game()
        if not game:
            logging.error("Game is not running. Cannot start bot.")
            window.status_label.setText("Status: Game not detected.")
            return

        Status.running = True
        logging.info("Bot started.")
        window.start_button.setText("Stop")
        window.status_label.setText("Status: Running")
    else:
        Status.running = False
        logging.info("Bot stopped.")
        window.start_button.setText("Start")
        window.status_label.setText("Status: Stopped")

# Function to toggle the chat mode (ALL vs TEAM)
def toggle_chat_mode():
    Status.chat_mode = "team" if Status.chat_mode == "all" else "all"
    window.chat_mode_label.setText(
        f"Chat Mode: {Status.chat_mode.capitalize()}")
    logging.info(f"Chat mode toggled to: {Status.chat_mode}")

# Function to toggle between the AI models
def toggle_model():
    if Status.ai_model == "openai":
        Status.ai_model = "gemini"
        logging.info("Switched to Gemini AI model.")
    elif Status.ai_model == "gemini":
        Status.ai_model = "openrouter"
        logging.info("Switched to OpenRouter AI model.")
    else:
        Status.ai_model = "openai"
        logging.info("Switched to OpenAI AI model.")

# Function to interact with Gemini API
def gemini_interact(user: str, message: str, chat_type: str):
    system_instruction = TEAM_CHAT_SYSTEM_PROMPT if chat_type == "team" else ALL_CHAT_SYSTEM_PROMPT
    user_message = f"Your team mate {user} wrote: {message}." if chat_type == "team" else f"Other player {user} wrote: {message}."

    genai.configure(api_key=GENAI_KEY) 

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        safety_settings=gemini_safety_settings,
        generation_config=gemini_generation_config,
        system_instruction=system_instruction,
    )
    
    logging.info(
        f"Interacting with Gemini API.\n- Chat type: {chat_type}\n- System prompt: {system_instruction}\n- Message: {user_message}")

    try:
        response = model.generate_content(user_message)
        reply = util.clean_text(response.text.replace('"', ''))
        logging.info(f"Gemini API response: {reply}")
        return reply

    except genai.exception.ApiException as e:
        logging.error(f"Gemini API Error: {e}")
        logging.error(f"Error Details: {e.message}, {e.errors}")
        return None
    except Exception as e: 
        logging.error(f"An unexpected error occurred: {e}")
        return None

# Function to interact with OpenAI API
def openai_interact(user: str, message: str, chat_type: str):
    system_instruction = TEAM_CHAT_SYSTEM_PROMPT if chat_type == "team" else ALL_CHAT_SYSTEM_PROMPT
    user_message = f"Your team-mate {user} wrote: {message}." if chat_type == "team" else f"Other player {user} wrote: {message}."

    OpenAI.api_key = OPENAI_KEY
    OpenAI.api_base = OPENAI_BASE_URL

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ]

    logging.info(
        f"Interacting with OpenAI API.\n- Chat type: {chat_type}\n- System prompt: {system_instruction}\n- Message: {user_message}")

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_KEY}"
        }
        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )
        reply = util.clean_text(response.choices[0].message.content)
        logging.info(f"OpenAI API response: {reply}")
        return reply
    except Exception as e:
        logging.error(f"An error occurred with OpenAI API: {e}")
        return None
    
# Function to interact with
def interact_openrouter(user: str, message: str, chat_type: str):
    system_instruction = TEAM_CHAT_SYSTEM_PROMPT if chat_type == "team" else ALL_CHAT_SYSTEM_PROMPT
    user_message = f"Your team-mate {user} wrote: {message}." if chat_type == "team" else f"Other player {user} wrote: {message}."

    OpenAI.api_key = OPENROUTER_KEY
    OpenAI.api_base = OPENROUTER_BASE_URL

    logging.info(
        f"Interacting with OpenAI API.\n- Chat type: {chat_type}\n- System prompt: {system_instruction}\n- Message: {user_message}")
    
    client = OpenAI(
        api_key = OPENROUTER_KEY,
        base_url = OPENROUTER_BASE_URL,
        default_headers = {
        "HTTP-Referer": "https://github.com/KristjanPikhof/CS2-Chatbot-with-GPT-Gemini-OpenRouter-integrations",
        "X-Title": "CS2 Assistant"},
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ]

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages
        )
        reply = util.clean_text(response.choices[0].message.content)
        logging.info(f"OpenRouter API response: {reply}")
        return reply
    except Exception as e:
        logging.error(f"An error occurred with OpenRouter API: {e}")
        return None

class AIThread(QThread):
    replyGenerated = pyqtSignal(str) 

    def __init__(self, user, message, chat_type, ai_model):
        super().__init__()
        self.user = user
        self.message = message
        self.chat_type = chat_type
        self.ai_model = ai_model

    def run(self):
        if self.ai_model == "openai":
            reply = openai_interact(self.user, self.message, self.chat_type)
        elif self.ai_model == "gemini":
            reply = gemini_interact(self.user, self.message, self.chat_type)
        elif self.ai_model == "openrouter":
            reply = interact_openrouter(self.user, self.message, self.chat_type)
        else:
            logging.error(f"Invalid AI model selected: {self.ai_model}")
            reply = None

        random_delay = random.uniform(2, 4)
        time.sleep(random_delay)
        self.replyGenerated.emit(reply)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.current_theme = "dark"
        self.setWindowTitle("Counter-Strike AI Assistant Chat-Bot 🤖")
        self.setWindowIcon(QIcon(resource_path('media/esmaabi_icon.ico')))
        self.setMinimumSize(450, 1025)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Title and detected game
        title_label = QLabel("Remember to always start the game first!")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        game = cp.detect_game()
        game_label = QLabel(f"Detected Game: {game if game else 'Not Detected'}")
        game_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(game_label)

        toggle_dark_mode_button = QPushButton("Dark Mode")
        toggle_dark_mode_button.clicked.connect(self.toggle_dark_mode)
        layout.addWidget(toggle_dark_mode_button)

        # Configuration Settings
        layout.addWidget(QLabel("Configuration Settings"))
        config_layout = QVBoxLayout()  # Initialize config_layout here!
        self.config_inputs = {}

        # Dictionary for custom titles
        titles = {
            'username': 'Username',
            'gameconlogpath': 'Console .log path',
            'chatkey': "ALL chat key",
            'teamchatkey': "TEAM chat key",
            'startstopkey': "Toggle Start/Stop key",
            'togglechatkey': "Toggle chat key",
            'openaiapikey': "OpenAI API key",
            'geminiapikey': "Gemini API key",
            'openrouterapikey': "OpenRouter API key",
            'openaimodel': "OpenAI model",
            'geminimodel': "Gemini model",
            'openroutermodel': "OpenRouter model",
            'allsystemprompt': "ALL chat prompt",
            'teamsystemprompt': "TEAM chat prompt",
            'togglemodelkey': "Toggle between AI models"
        }

        # Dictionary for custom hints
        hints = {
            'username': 'Enter usernames separated by "; ". Example: User1; User2',
            'gameconlogpath': 'Enter the path to ...\game\csgo\console.log',
            'chatkey': "Enter the chat keybind (default 'y')",
            'teamchatkey': "Enter the team chat keybind (default 'u')",
            'startstopkey': "Enter the start/stop keybind (default 'F10')",
            'togglechatkey': "Enter the toggle chat keybind (default 'F11')",
            'openaiapikey': "Enter your OpenAI API key",
            'geminiapikey': "Enter your Gemini API key",
            'openrouterapikey': "Enter your OpenRouter API key",
            'openaimodel': "Enter OpenAI model. Example: gpt-4o-mini",
            'geminimodel': "Enter Gemini model. Example: gemini-1.5-flash-latest",
            'openroutermodel': "Enter OpenRouter model. Example: openai/gpt-4o-mini",
            'allsystemprompt': "Enter your AI system prompt for ALL chat",
            'teamsystemprompt': "Enter your AI system prompt for TEAM chat",
            'togglemodelkey': "Enter the keybind to toggle between AI models",
        }

        for key in config['SETTINGS']:
            h_layout = QHBoxLayout()
            label = QLabel(f"{titles.get(key, key.replace('_', ' ').title())}:")
            if key in ("openaiapikey", "geminiapikey", "openrouterapikey"):
                input_field = QLineEdit(config['SETTINGS'][key])
                input_field.setEchoMode(QLineEdit.Password)
                input_field.setToolTip(hints.get(key, ""))
                input_field.setPlaceholderText(hints.get(key, ""))
            elif key == 'gameconlogpath':
                input_field = QLineEdit(config['SETTINGS'][key])
                input_field.setToolTip(hints.get(key, ""))
                input_field.setPlaceholderText(hints.get(key, ""))
                browse_button = QPushButton("Browse")
                browse_button.clicked.connect(lambda _, key=key: self.browse_file(key))
                h_layout.addWidget(browse_button)
            else:
                input_field = QLineEdit(config['SETTINGS'][key])
                input_field.setToolTip(hints.get(key, ""))
                input_field.setPlaceholderText(hints.get(key, ""))
            self.config_inputs[key] = input_field
            h_layout.addWidget(label)
            h_layout.addWidget(input_field)
            config_layout.addLayout(h_layout)

        layout.addLayout(config_layout)

        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_config)
        layout.addWidget(save_button)

        # Bot Control
        layout.addWidget(QLabel("Bot Control"))
        self.status_label = QLabel("Status: Stopped")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(toggle_status)
        layout.addWidget(self.start_button)

        # Chat Mode
        layout.addWidget(QLabel("Chat Mode"))
        self.chat_mode_label = QLabel(f"Chat Mode: {Status.chat_mode.capitalize()}")
        self.chat_mode_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.chat_mode_label)

        toggle_chat_button = QPushButton("Toggle Chat Mode")
        toggle_chat_button.clicked.connect(toggle_chat_mode)
        layout.addWidget(toggle_chat_button)

        # AI Model Selection
        layout.addWidget(QLabel("Select AI Model"))
        ai_group = QButtonGroup(self)
        self.openai_radio = QRadioButton("OpenAI")
        self.gemini_radio = QRadioButton("Gemini")
        self.openrouter_radio = QRadioButton("OpenRouter")
        ai_group.addButton(self.openai_radio)
        ai_group.addButton(self.gemini_radio)
        ai_group.addButton(self.openrouter_radio)
        layout.addWidget(self.openai_radio)
        layout.addWidget(self.gemini_radio)
        layout.addWidget(self.openrouter_radio)

        if Status.ai_model == "openai":
            self.openrouter_radio.setChecked(False)
            self.openai_radio.setChecked(True)
        elif Status.ai_model == "gemini":
            self.openai_radio.setChecked(False)
            self.gemini_radio.setChecked(True)
        elif Status.ai_model == "openrouter":
            self.gemini_radio.setChecked(False)
            self.openrouter_radio.setChecked(True)            
        else:
            logging.error(f"Error changing AI model: {Status.ai_model}")
            
        ai_group.buttonClicked.connect(self.set_ai_model)

        # Credits
        credits_label = QLabel(
            "Rewritten by Esmaabi. Inspired from xsync3d & Isaac-Duarte")
        credits_label.setAlignment(Qt.AlignRight)
        layout.addWidget(credits_label)

        github_label = QLabel('<a style="text-decoration: none; color: #0078D7;" href="https://github.com/KristjanPikhof">GitHub: github.com/KristjanPikhof</a>') 
        github_label.setAlignment(Qt.AlignRight)
        github_label.setOpenExternalLinks(True)
        layout.addWidget(github_label) 

        self.setLayout(layout)

    def process_message(self, parsed_log):
        """Processes a single message from the log using a separate thread."""
        self.ai_thread = AIThread(parsed_log['username'], parsed_log['message'], Status.chat_mode, Status.ai_model)
        self.ai_thread.replyGenerated.connect(self.send_reply)
        self.ai_thread.start()

    def send_reply(self, reply):
         """Sends the reply to the game."""
         if reply:
             cp.sim_key_presses(reply, chat_type=Status.chat_mode)
         else:
             logging.warning("No reply to send.")

    def toggle_dark_mode(self):
       """Toggles between light and dark mode."""
       if self.current_theme == "dark":
           QApplication.instance().setStyleSheet(LIGHT_MODE_STYLESHEET)
           self.current_theme = "light"
           self.sender().setText("Dark Mode")
       else:
           QApplication.instance().setStyleSheet(DARK_MODE_STYLESHEET)
           self.current_theme = "dark"
           self.sender().setText("Light Mode")

    def browse_file(self, key):
        """Opens a file dialog to browse for the console.log file."""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Log files (*.log)")
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]
            self.config_inputs[key].setText(selected_file)

    def save_config(self):
        """Saves the configuration settings."""
        try:
            for key in config['SETTINGS']:
                config['SETTINGS'][key] = self.config_inputs[key].text()

            config_path = resource_path(CONFIG_FILE)
            print("Trying to save config to:", config_path)

            with open(config_path, 'w', encoding='utf-8') as configfile:
                config.write(configfile)

            # Update global variables
            global BLACKLISTED_USERNAMES, CON_LOG_FILE_PATH, CHAT_KEY, TEAM_CHAT_KEY
            global START_STOP_KEY, TOGGLE_CHAT_KEY, ALL_CHAT_SYSTEM_PROMPT, TEAM_CHAT_SYSTEM_PROMPT
            global OPENAI_KEY, OPENROUTER_KEY, GENAI_KEY, TOGGLE_MODEL_KEY
            BLACKLISTED_USERNAMES = get_blacklisted_usernames()
            CON_LOG_FILE_PATH = config['SETTINGS']['gameconlogpath']
            CHAT_KEY = config['SETTINGS']['chatkey']
            TEAM_CHAT_KEY = config['SETTINGS']['teamchatkey']
            START_STOP_KEY = config['SETTINGS']['startstopkey']
            TOGGLE_CHAT_KEY = config['SETTINGS']['togglechatkey']
            TOGGLE_MODEL_KEY = config['SETTINGS']['togglemodelkey']
            ALL_CHAT_SYSTEM_PROMPT = config['SETTINGS']['allsystemprompt']
            TEAM_CHAT_SYSTEM_PROMPT = config['SETTINGS']['teamsystemprompt']

            OPENAI_KEY = config['SETTINGS']['openaiapikey']
            OPENROUTER_KEY = config['SETTINGS']['openrouterapikey']
            GENAI_KEY = config['SETTINGS']['geminiapikey']


            # Update UI elements
            self.config_inputs['username'].setText(config['SETTINGS']['username'])
            self.config_inputs['gameconlogpath'].setText(config['SETTINGS']['gameconlogpath'])
            self.config_inputs['chatkey'].setText(config['SETTINGS']['chatkey'])
            self.config_inputs['teamchatkey'].setText(config['SETTINGS']['teamchatkey'])
            self.config_inputs['startstopkey'].setText(config['SETTINGS']['startstopkey'])
            self.config_inputs['togglechatkey'].setText(config['SETTINGS']['togglechatkey'])
            self.config_inputs['openaiapikey'].setText(config['SETTINGS']['openaiapikey'])
            self.config_inputs['geminiapikey'].setText(config['SETTINGS']['geminiapikey'])
            self.config_inputs['openaimodel'].setText(config['SETTINGS']['openaimodel'])
            self.config_inputs['geminimodel'].setText(config['SETTINGS']['geminimodel'])
            self.config_inputs['allsystemprompt'].setText(config['SETTINGS']['allsystemprompt'])
            self.config_inputs['teamsystemprompt'].setText(config['SETTINGS']['teamsystemprompt'])
            self.config_inputs['openrouterapikey'].setText(config['SETTINGS']['openrouterapikey'])
            self.config_inputs['openroutermodel'].setText(config['SETTINGS']['openroutermodel'])
            self.config_inputs['togglemodelkey'].setText(config['SETTINGS']['togglemodelkey'])

            # Reload the configuration
            load_config()

            # Restart the application
            QMessageBox.information(self, "Applying settings", "Configuration settings saved successfully. The application will now restart to apply the new settings.")
            python = sys.executable
            os.execl(python, python, *sys.argv)

        except Exception as e:
            logging.error(f"Error saving config file: {e}")
            QMessageBox.critical(self, "Error",
                                 f"An error occurred while saving the settings: {e}")

    def set_ai_model(self, button):
        ai_model = button.text().lower()
        Status.ai_model = ai_model

        if ai_model == "openai":
            logging.info("Switched to OpenAI AI model. ")
        elif ai_model == "gemini":
            logging.info("Switched to Gemini AI model. ")
        elif ai_model == "openrouter":
            logging.info("Switched to OpenRouter AI model. ")
        else:
            logging.error(f"Error changing AI model: {ai_model}")
            return

        self.update_ui_for_ai_model()

    def update_ui_for_ai_model(self):
        """Update UI elements based on the selected AI model."""
        if Status.ai_model == "openai":
            self.openrouter_radio.setChecked(False)
            self.openai_radio.setChecked(True)
        elif Status.ai_model == "gemini":
            self.openrouter_radio.setChecked(False)
            self.gemini_radio.setChecked(True)
        elif Status.ai_model == "openrouter":
            self.gemini_radio.setChecked(False)
            self.openrouter_radio.setChecked(True)
        else:
            logging.error(f"Error changing AI model: {Status.ai_model}")


    def closeEvent(self, event):
        """Handles the close event for the main window."""
        cleanup()
        event.accept()

def check_log_file():
    """Checks the log file for new messages and processes them."""
    global logfile, game  # Access global variables

    if Status.running and logfile:
        line = cp.rt_file_read(logfile)
        if not line:
            return

        parsed_log = cp.parse_log(game, line)

        if parsed_log and parsed_log['username'] and parsed_log['message']:
            if cp.should_respond(parsed_log, Status.chat_mode):
                window.process_message(parsed_log)  


def process_message(parsed_log):
    """Processes a single message from the log."""
    if Status.ai_model == "openai":
        reply = openai_interact(parsed_log['username'],
                                parsed_log['message'],
                                Status.chat_mode)
    elif Status.ai_model == "gemini":
        reply = gemini_interact(parsed_log['username'],
                                parsed_log['message'],
                                Status.chat_mode)
    elif Status.ai_model == "openrouter":
        reply = interact_openrouter(parsed_log['username'],
                                parsed_log['message'],
                                Status.chat_mode)    
    else:
        logging.error(f"Invalid AI model selected: {Status.ai_model}")
        reply = None

    if reply:
        cp.sim_key_presses(reply, chat_type=Status.chat_mode)        

def cleanup():
    """Performs cleanup tasks before the application exits."""
    global logfile
    logging.info("Cleaning up and exiting...")
    logging.info("Good Luck & Have a Fun!")
    logging.info("GitHub: github.com/KristjanPikhof")
    if Status.running:
        toggle_status()
    if logfile:
        logfile.close()  

def main():
    global window, logfile, game
    load_config()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    QApplication.instance().setStyleSheet(DARK_MODE_STYLESHEET)

    try:
        logfile = None
        game = cp.detect_game()
        logging.info(f"Detected game: {game}")

        register_hotkeys()

        if config['SETTINGS']['gameconlogpath']:
            try:
                logfile = open(CON_LOG_FILE_PATH, encoding='utf-8')
                logfile.seek(0, 2) 
                logging.info(f"Opened logfile: {CON_LOG_FILE_PATH}")
            except FileNotFoundError:
                logging.error(f"Log file not found: {CON_LOG_FILE_PATH}")
                logfile = None
        else:
            logging.warning(
                "gameconlogpath is not set in config.ini. Common path: \nC:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\console.log")
            logfile = None
            
        # Timer to check the log file periodically
        timer = QTimer()
        timer.timeout.connect(check_log_file)  
        timer.start(50)  # Check every 50 milliseconds

        sys.exit(app.exec_()) # Start the Qt event loop
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()
