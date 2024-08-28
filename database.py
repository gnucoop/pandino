import sqlite3
import sys
from cryptography.fernet import Fernet
import os

# Generate a key for encryption and decryption
# Store the key in an environment variable or a secure file
KEY = os.environ.get('ENCRYPTION_KEY')
if not KEY:
    KEY = Fernet.generate_key()
    print(f"Generated new encryption key: {KEY.decode()}. Please store this securely.")
else:
    print(f"Using existing ENCRYPTION_KEY: {KEY}")
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
    print(f"Connecting to database: users.db")
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    encrypted_api_key = cipher_suite.encrypt(api_key.encode())
    print(f"Encrypted API key: {encrypted_api_key}")
    try:
        print(f"Attempting to add user: {username}")
        cursor.execute('INSERT INTO users (username, api_key) VALUES (?, ?)', (username, encrypted_api_key))
        conn.commit()
        print(f"User {username} added successfully.")
    except sqlite3.IntegrityError as e:
        print(f"Error: {e}")
        print(f"User {username} or API key already exists.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        conn.close()
        print("Database connection closed.")

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
    cursor.execute('SELECT * FROM users WHERE api_key = ?', (api_key,))
    user = cursor.fetchone()
    conn.close()
    return user is not None

if __name__ == "__main__":
    init_db()
    if len(sys.argv) > 1:
        if sys.argv[1] == "add_user" and len(sys.argv) == 4:
            username, api_key = sys.argv[2], sys.argv[3]
            add_user(username, api_key)
        elif sys.argv[1] == "list_users":
            list_users()
        else:
            print("Usage: python database.py [add_user <username> <api_key> | list_users]")
    else:
        print("Usage: python database.py [add_user <username> <api_key> | list_users]")
