#!/usr/bin/env python3

from multiprocessing.connection import Listener
import os
import atexit

from llm_research.engine import LLMEngine

print("Modules loaded.")


SOCKET = "/tmp/llm_chat.sock"
if os.path.exists(SOCKET):
    os.remove(SOCKET)

# Start listener
listener = Listener(SOCKET, family="AF_UNIX")

# Set up cleanup for server termination
def cleanup(listener):
    print("Stopping server.")
    listener.close()

    if os.path.exists(SOCKET):
        os.remove(SOCKET)
    print("Server closed.")

atexit.register(cleanup, listener)

# Start model
print("Loading model...")
model_name = "Qwen/Qwen3-0.6B-Base"
engine = LLMEngine(model_name)
print("Done loading model.")

print("Testing model...")
response = engine.complete("The color of the moon is:")
print(response)
engine.reset_session()
print("Done testing model.")


while True:
    print("Waiting for client...")
    conn = listener.accept()
    print("Client connected.")

    try:
        while True:
            try:
                message = conn.recv()
            except EOFError:
                print("Client disconnected unexpectedly.")
                break

            if message == ("disconnect", None):
                print("Client disconnected.")
                break

            prompt = message

            print(f"Prompt: {prompt}")

            response = engine.stream_chat(prompt)

            for chunk in response:
                print(chunk, end="", flush=True)
                conn.send(("chunk", chunk))

            conn.send(("end", None))
            print("\nSent response to client.")

    except (BrokenPipeError, ConnectionResetError):
        print("\nClient connection lost.")

    finally:
        conn.close()

        # Reset per-client/session variables here
        engine.reset_session() 

        print("Session reset.")


print("Server closed.")




