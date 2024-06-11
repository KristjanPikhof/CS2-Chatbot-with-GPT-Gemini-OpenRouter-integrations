import time
import configparser
import keyboard
import psutil
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s: %(message)s', datefmt = '%d.%m.%Y %H:%M:%S')

# Initialize the config parser and read the config file
config = configparser.ConfigParser()
CONFIG_FILE = 'config.ini'

def resource_path(relative_path):
    """Get the correct path to a resource file, even when running as executable."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

try:
    config.read(resource_path(CONFIG_FILE), encoding='utf-8')
except Exception as e:
    logging.error(f"Error reading config file: {e}")

# Load settings from the config file with defaults
# Split the blacklisted usernames by semicolon and strip whitespace
BLACKLISTED_USERNAMES = [username.strip() for username in config['SETTINGS'].get('username', '').replace('[DEAD]', '').split(';')]

CON_LOG_FILE_PATH = config['SETTINGS'].get('gameconlogpath', 'C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\console.log')
CHAT_KEY = config['SETTINGS'].get('chatkey', 'y')
TEAM_CHAT_KEY = config['SETTINGS'].get('teamchatkey', 'u')

def detect_game(custom_proc="customproc"):
    """
    Detects the game process that is running and returns its name.

    Args:
        custom_proc (str): Custom process name to search for (default is "customproc")

    Returns:
        str: The detected game process name or None if not found
    """
    known_games = {"hl.exe": "hl", "hl2.exe": "hl2", "cs2.exe": "cs2", custom_proc: custom_proc.strip(".exe")}
    for proc in psutil.process_iter():
        try:
            name = proc.name().lower()
            if name in known_games:
                return known_games[name]
            return "cs2"
        except psutil.NoSuchProcess:
            continue
    #return None
    return "cs2"

def parse_log(game, line: str):
    """
    Parses source console logs, detecting chat messages.

    Args:
        game (str): Specifies the game to use the appropriate format
        line (str): The string fetched from the game console log

    Returns:
        dict: Dictionary containing parsed log information
    """
    if "Source2Shutdown" in line:
        exit()

    parsed_log = {"username": "", "message": "", "team": "", "chat_type": ""}

    # Match based on game type
    match game:
        case "cs2":
            if "[ALL]" in line:
                parsed_log["chat_type"] = "all"
                parsed_data = line.partition("[ALL] ")[2].split(": ")
                if len(parsed_data) == 2:
                    parsed_log["username"] = parsed_data[0]
                    parsed_log["message"] = parsed_data[1]
            elif "[CT]" in line or "[T]" in line:
                parsed_log["chat_type"] = "team"
                parsed_data = line.partition("] ")[2].split(": ")
                if len(parsed_data) == 2:
                    name_and_location = parsed_data[0].rsplit("﹫", 1)
                    parsed_log["username"] = name_and_location[0].replace("[DEAD]", '').strip()
                    parsed_log["message"] = parsed_data[1]
                
                # Identify team
                if "[CT]" in line:
                    parsed_log["team"] = "CT"
                elif "[T]" in line:
                    parsed_log["team"] = "T"
            elif "[DEAD]" in line:
                parsed_log["chat_type"] = "dead"
                parsed_data = line.partition("[DEAD] ")[2].split(": ")
                if len(parsed_data) == 2:
                    parsed_log["username"] = parsed_data[0].replace(" [DEAD]", '')
                    parsed_log["message"] = parsed_data[1]
                
        case "hl":
            if ": " in line:
                parsed_data = line.split(": ")
                parsed_log["username"] = parsed_data[0][1:]  # Strip weird unicode character
                parsed_log["message"] = parsed_data[1]
                parsed_log["chat_type"] = "all"
        
        case "hl2":
            if "*DEAD*" in line:
                parsed_data = line.replace("*DEAD* ", '').split(" : ")
                parsed_log["username"] = parsed_data[0]
                parsed_log["message"] = parsed_data[1] if len(parsed_data) > 1 else ""
                parsed_log["chat_type"] = "dead"
            elif " : " in line:
                parsed_data = line.split(" : ")
                parsed_log["username"] = parsed_data[0]
                parsed_log["message"] = parsed_data[1]
        
        case _:
            return None
    
    parsed_log["username"] = parsed_log["username"].replace(u'\u200e', '')  # Clean up username
    return parsed_log

def rt_file_read(file):
    """
    Reads console.log in real-time.

    Args:
        file (file): The file object to read from

    Returns:
        str: The read line from the file
    """
    line = file.readline()
    return line

def sim_key_presses(text: str, delay=0.01, chat_type="all"):
    """
    Simulates key presses to send a message in-game.

    Args:
        text (str): The text message to be sent
        delay (float): Delay between key presses
        chat_type (str): The type of chat to send the message to (team or all)
    """
    try:
        if chat_type == "team":
            keyboard.press_and_release(TEAM_CHAT_KEY)
        else:
            keyboard.press_and_release(CHAT_KEY)
        time.sleep(delay)
        encoded_text = text.encode('unicode_escape').decode('utf-8')
        keyboard.write(encoded_text)
        time.sleep(delay)
        keyboard.press_and_release('enter')
        time.sleep(0.15)
    except Exception as e:
        logging.error(f"Error in sim_key_presses: {e}")

def should_respond(parsed_log, current_mode):
    """
    Determine if the message should be responded to based on teammate logic and current chat mode.
    
    Args:
        parsed_log (dict): The parsed log information
        current_mode (str): The current chat mode ("all" or "team")
    
    Returns:
        bool: True if the message should be responded to, False otherwise
    """
    sanitized_username = parsed_log.get('username', '').replace("[DEAD]", '').strip()
    if sanitized_username in BLACKLISTED_USERNAMES:
        logging.info(f"Ignoring message from blacklisted user: {sanitized_username}")
        return False
    if current_mode == "team" and parsed_log['chat_type'] != "team":
        return False
    if current_mode == "all" and parsed_log['chat_type'] != "all":
        return False
    return True
