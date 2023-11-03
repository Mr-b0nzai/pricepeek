import sqlite3
import hashlib
import socket
import threading


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.bind(('198.199.86.77', 4567)) # uncomment this line to run on a public server
server.bind(('localhost', 4567))

server.listen()

conn = sqlite3.connect('users.db')
cur = conn.cursor()


# FOR TESTING PURPOSES ONLY
cur.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username VARCHAR(255) NOT NULL, password VARCHAR(255) NOT NULL)')

username1, password1 = 'admin', hashlib.sha256('admin'.encode()).hexdigest()
username2, password2 = 'john', hashlib.sha256('pass'.encode()).hexdigest()
username3, password3 = 'doe', hashlib.sha256('blah'.encode()).hexdigest()
username4, password4 = 'jojo', hashlib.sha256('midget'.encode()).hexdigest()
cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username1, password1))
cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username2, password2))
cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username3, password3))
cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username4, password4))

conn.commit()

def handle_client(c):
    try:
        username = c.recv(1024).decode()
        password = c.recv(1024).decode()
        password = hashlib.sha256(password.encode()).hexdigest()

        conn = sqlite3.connect('users.db')
        cur = conn.cursor()

        cur.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))

        if cur.fetchall():
            c.send('Login successful'.encode())
            print(f'{username} logged in')
            # do stuff here afer login
            
            
        else:
            c.send('Login failed'.encode())
            print(f'{username} failed to log in')
    except ConnectionResetError:
        print(f'Connection reset by {c.getpeername()}')
    
    finally:
        c.close()
    
while True:
    c, addr = server.accept()
    print(f'Connected to {addr}')
    t = threading.Thread(target=handle_client, args=(c,))
    t.start()
    
