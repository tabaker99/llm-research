
from multiprocessing.connection import Client

import signal
import sys

import atexit

SOCKET = "/tmp/llm_chat.sock"


global running_state
running_state = True

n_messages = 0

GREEN = "\033[32m"
COLOR2 = "\033[36m"
RESET = "\033[0m"


def frame_prompt(prompt):
    frame = "<PROMPT>{}</PROMPT><RESPONSE>"
    framed_prompt = frame.format(prompt)
    return framed_prompt

def sigint_handler(sig, frame):
    print("Stopping client...")
    global running_state
    if (running_state == False):
        try:
            conn.close()
            exit(0)
        except:
            exit(1)
    else:
        running_state = False

def cleanup(conn):
    try:
        conn.send(("disconnect", None))
    except (BrokenPipeError, EOFError, OSError):
        pass

    conn.close()

# Connect to socket
conn = Client(SOCKET, family="AF_UNIX")

atexit.register(cleanup, conn)


while(running_state):
    prompt = input(f"{GREEN}USER>>{RESET}")
    #framed_prompt = frame_prompt(prompt)

    conn.send(prompt)
    
    receiving = True
    response = ""
    print(f"\n{COLOR2}", end="")
    while receiving and running_state:
        message_type, message = conn.recv()
        if message_type == "end":
            receiving = False
        else:
            response = response + message
            print(message, end="", flush=True)

    print(f"{RESET}")


conn.close()
print("\nClient connection closed.\n")
