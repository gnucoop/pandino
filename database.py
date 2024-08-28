import sqlite3
import sys
from cryptography.fernet import Fernet, InvalidToken
import os
import base64

# Generate a key for encryption and decryption
# Store the key in an environment variable or a secure file
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

KEY = os.environ.get('ENCRYPTION_KEY')
if not KEY:
    print("Error: ENCRYPTION_KEY not found in environment variables.")
    sys.exit(1)

try:
    # Ensure the key is properly formatted
    KEY = base64.urlsafe_b64encode(base64.urlsafe_b64decode(KEY + '=' * (-len(KEY) % 4)))
    print("Using ENCRYPTION_KEY from environment variables.")
except Exception as e:
    print(f"Error with ENCRYPTION_KEY: {e}")
    sys.exit(1)

cipher_suite = Fernet(KEY)

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            api_key TEXT NOT NULL UNIQUE
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def add_user(username, api_key):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    encrypted_api_key = cipher_suite.encrypt(api_key.encode())
    try:
        cursor.execute('INSERT INTO users (username, api_key) VALUES (?, ?)', (username, encrypted_api_key))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    except Exception:
        pass
    finally:
        conn.close()

def list_users():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT username FROM users')
    users = cursor.fetchall()
    conn.close()
    if users:
        print("Existing users:")
        for user in users:
            print(user[0])
    else:
        print("No users found in the database.")

def validate_api_key(api_key):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT api_key FROM users')
    encrypted_keys = cursor.fetchall()
    conn.close()
    
    for (encrypted_key,) in encrypted_keys:
        try:
            if isinstance(encrypted_key, str):
                encrypted_key = encrypted_key.encode()
            decrypted_key = cipher_suite.decrypt(encrypted_key).decode().strip()
            if decrypted_key == api_key.strip():
                return True, "API key match found"
        except InvalidToken:
            pass
        except Exception:
            pass
    return False, "No matching API key found"

def print_stored_keys():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT username, api_key FROM users')
    users = cursor.fetchall()
    conn.close()
    print("Stored API keys:")
    for username, encrypted_key in users:
        print(f"Username: {username}, Encrypted key: {encrypted_key}")
        try:
            if isinstance(encrypted_key, str):
                encrypted_key = encrypted_key.encode()
            decrypted_key = cipher_suite.decrypt(encrypted_key).decode()
            print(f"  Decrypted key: {decrypted_key}")
        except Exception as e:
            print(f"  Error decrypting key: {str(e)}")

def print_help():
    print("Usage: python database.py <command>")
    print("Commands:")
    print("  init_db                     Initialize the database")
    print("  add_user <username> <api_key>  Add a new user")
    print("  list_users                  List all users")
    print("  print_keys                  Print all stored API keys")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "init_db":
            init_db()
        elif sys.argv[1] == "add_user" and len(sys.argv) == 4:
            username, api_key = sys.argv[2], sys.argv[3]
            add_user(username, api_key)
        elif sys.argv[1] == "list_users":
            list_users()
        elif sys.argv[1] == "print_keys":
            print_stored_keys()
        else:
            print_help()
    else:
        print_help()
