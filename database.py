import sqlite3

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

def add_user(username, api_key):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, api_key) VALUES (?, ?)', (username, api_key))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"User {username} or API key already exists.")
    conn.close()

def validate_api_key(api_key):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE api_key = ?', (api_key,))
    user = cursor.fetchone()
    conn.close()
    return user is not None
