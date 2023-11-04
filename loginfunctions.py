import time
from tkinter import *
import socket
import threading


is_logged_in = True


def send_credentials(username, password):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client.connect(('198.199.86.77', 4567)) # uncomment this line to run on a public server
    client.connect(('localhost', 4567))
    def send_thread():
        global is_logged_in
        while not is_logged_in:
            try:
                client.send(username.encode())
                client.send(password.encode())

                result = client.recv(1024).decode()
                print(result)
                if result == 'Login successful':
                    is_logged_in = True
                    print(is_logged_in)
                    return True
                else:
                    return False
            except ConnectionAbortedError:
                print('Connection aborted, retrying in 1 second...')
                time.sleep(1)
                
    t = threading.Thread(target=send_thread, daemon=True)
    t.start()
    t.join(timeout=3)
    return is_logged_in

# def on_login_complete(result):
#     if result == 'Login successful':
#         # login_success = customtkinter.CTkLabel(master=frame, text='Login successful!', font=("Arial Bold", 20))
#         # login_success.pack()
#         print(result)
#     else:
#         # login_success = customtkinter.CTkLabel(master=frame, text='Login failed!', font=("Arial Bold", 20))
#         # login_success.pack()
#         print(result)

# def login(username, password):
#     # Run send_credentials in a separate thread
#     t = threading.Thread(target=send_credentials, args=(username, password), daemon=True)
#     t.start()
#     t.join(timeout=3)  # Wait for the thread to complete for 1 second
#     print(is_logged_in)
#     if t.is_alive():
#         # The thread is still running, so assume the login failed
#         on_login_complete('failed')
#     else:
#         # The thread has completed, so get the result and update the UI accordingly
#         result = send_credentials(username, password)
#         on_login_complete(result)
    
# root = customtkinter.CTk()
# root.geometry("500x500")

# customtkinter.set_appearance_mode('dark')
# customtkinter.set_default_color_theme('dark-blue')

# frame = customtkinter.CTkFrame(master=root)
# frame.pack(pady=20, padx=60, fill='both', expand=True)

# root.title("PricePeek")

# login_lable = customtkinter.CTkLabel(master=frame, text='Login to PricePeek', font=("Arial Bold", 20))
# login_lable.pack()

# username = customtkinter.CTkLabel(master=frame, text='Username', font=("Arial Bold", 15))
# username.pack()
# username_entry = customtkinter.CTkEntry(frame, placeholder_text='username')
# username_entry.pack()

# password = customtkinter.CTkLabel(master=frame, text='Password', font=("Arial Bold", 15))
# password.pack()
# password_entry = customtkinter.CTkEntry(frame, placeholder_text='password', show='*')
# password_entry.pack()

# login_btn = customtkinter.CTkButton(master=frame, text='Login', command=login)
# login_btn.pack()

print(is_logged_in)


    

# conn = sqlite3.connect('users.db')
# cur = conn.cursor()


# # FOR TESTING PURPOSES ONLY
# cur.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username VARCHAR(255) NOT NULL, password VARCHAR(255) NOT NULL)')

# username1, password1 = 'admin', hashlib.sha256('admin'.encode()).hexdigest()
# username2, password2 = 'john', hashlib.sha256('pass'.encode()).hexdigest()
# username3, password3 = 'doe', hashlib.sha256('blah'.encode()).hexdigest()
# username4, password4 = 'jojo', hashlib.sha256('midget'.encode()).hexdigest()
# cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username1, password1))
# cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username2, password2))
# cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username3, password3))
# cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username4, password4))

# conn.commit()

# root.configure(background='#060e2e')


# def on_closing():
#     # add any cleanup code here
#     root.destroy()
#     sys.exit()

# root.protocol("WM_DELETE_WINDOW", on_closing)

# root.mainloop()
