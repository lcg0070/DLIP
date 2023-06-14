import socket
import pickle

def start_server():
    host = 'localhost' 
    port = 12345  
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        conn, addr = s.accept()

        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                array_received = pickle.loads(data)
                print('Received from client:', array_received)

if __name__ == '__main__':
    start_server()
