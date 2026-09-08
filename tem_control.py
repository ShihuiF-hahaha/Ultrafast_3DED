# tem_control_server.py
# Run this script on the TEM computer.

import socket
import traceback
import time

HOST = "0.0.0.0"
PORT = 9090

tem = None


def init_tem():
    """Initialize Instamatic TEM controller."""
    global tem

    if tem is None:
        import instamatic.TEMController as TC
        tem = TC.initialize()

    return tem


def get_alpha():
    """Return current alpha tilt angle."""
    t = init_tem()
    return float(t.stage.a)


def set_alpha(angle):
    """Set alpha tilt angle."""
    t = init_tem()
    angle = float(angle)

    t.stage.set(a=angle)
    time.sleep(2)

    return get_alpha()


def stop_stage():
    """Stop stage motion."""
    t = init_tem()
    t.stage.stop()
    return get_alpha()


def handle_command(command):
    command = command.strip()
    parts = command.split()

    if not parts:
        return "ERR empty command"

    cmd = parts[0].upper()

    if cmd == "PING":
        return "OK PONG"

    if cmd == "GET_ALPHA":
        alpha = get_alpha()
        return f"OK ALPHA {alpha:.4f}"

    if cmd == "TILT":
        if len(parts) != 2:
            return "ERR usage: TILT <angle>"
        alpha = set_alpha(parts[1])
        return f"OK TILTED {alpha:.4f}"

    if cmd == "STOP":
        alpha = stop_stage()
        return f"OK STOPPED {alpha:.4f}"

    return f"ERR unknown command: {command}"


def main():
    # Important: initialize TEM only inside main(), not at top level.
    init_tem()

    print(f"TEM control server listening on {HOST}:{PORT}")
    print("Available commands: PING, GET_ALPHA, TILT <angle>, STOP")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        while True:
            conn, addr = server.accept()

            with conn:
                try:
                    data = conn.recv(1024).decode("utf-8").strip()
                    print(f"From {addr}: {data}")

                    response = handle_command(data)
                    conn.sendall((response + "\n").encode("utf-8"))

                    print("Response:", response)

                except Exception:
                    err = traceback.format_exc()
                    print(err)
                    conn.sendall(
                        ("ERR " + err.replace("\n", " | ") + "\n").encode("utf-8")
                    )


if __name__ == "__main__":
    main()
