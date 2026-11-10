"""Explicit XDG GlobalShortcuts setup, per official portal and shortcuts specs.

Standalone GI helper owns the portal session; no mandatory GI dependency in uv.
Consent belongs to the compositor. Never fake success if portal is unavailable.
"""

import importlib.util
import json
import os
import select
import subprocess
import sys
import time
import uuid
from pathlib import Path


def preferred_trigger(mask, key):
    if key.startswith("button"):
        raise ValueError(
            "Portal supports keyboard shortcuts only; use X11 or desktop mouse bindings"
        )
    modifiers = [
        name for bit, name in ((4, "CTRL"), (8, "ALT"), (1, "SHIFT"), (64, "LOGO")) if mask & bit
    ]
    return "+".join([*modifiers, key])


def launch():
    from .config import Config
    from .hotkeys import Activation, bindings
    from .processes import python_command
    from .service import request

    config = Config.load()

    def configured_entries(value):
        return [
            {"id": f"{action}_{index}", "action": action, "trigger": preferred_trigger(mask, key)}
            for index, (action, mask, key) in enumerate(bindings(value))
        ]

    args = [*python_command("mouthkeys.portal_shortcuts"), "--helper"]
    if importlib.util.find_spec("gi") is None and Path("/usr/bin/python3").is_file():
        args = ["/usr/bin/python3", "-I", str(Path(__file__).resolve()), "--helper"]
    activation = Activation(config.hotkey_mode, request)
    while config.shortcut_backend == "portal":
        entries = configured_entries(config)
        child = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        changed = False
        checked_at = 0
        buffer = b""
        try:
            child.stdin.write((json.dumps(entries) + "\n").encode())
            child.stdin.close()
            while True:
                ready = select.select([child.stdout], [], [], 0.25)[0]
                if ready:
                    chunk = os.read(child.stdout.fileno(), 4096)
                    if not chunk:
                        break
                    buffer += chunk
                    if len(buffer) > 65536:
                        raise RuntimeError("Oversized shortcut event")
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        value = json.loads(line)
                        if "error" in value:
                            raise RuntimeError(value["error"])
                        activation.event(value["action"], value["down"], value["at"], value["id"])
                if time.monotonic() - checked_at >= 1:
                    checked_at = time.monotonic()
                    refreshed = Config.load()
                    changed = refreshed.shortcut_backend != "portal" or (
                        not activation.pressed
                        and (
                            configured_entries(refreshed) != entries
                            or refreshed.hotkey_mode != config.hotkey_mode
                        )
                    )
                    if changed:
                        config = refreshed
                        activation.mode = config.hotkey_mode
                        break
            if not changed:
                if child.wait(timeout=2):
                    raise RuntimeError("GlobalShortcuts portal unavailable, cancelled, or denied")
                return 0
        finally:
            activation.finish()
            child.stdout.close()
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)
    return 0


def helper():
    import gi

    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib

    entries = json.loads(sys.stdin.readline(65536))
    by_id = {entry["id"]: entry["action"] for entry in entries}
    connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    loop = GLib.MainLoop()
    pending = {}
    state = {"session": None, "ready": False}
    interface = "org.freedesktop.portal.GlobalShortcuts"
    destination = "org.freedesktop.portal.Desktop"
    path = "/org/freedesktop/portal/desktop"

    def fail(message):
        print(json.dumps({"error": message}), flush=True)
        loop.quit()

    def request(method, signature, parameters, token, callback):
        sender = connection.get_unique_name()[1:].replace(".", "_")
        expected = path + "/request/" + sender + "/" + token
        pending[expected] = callback
        result = connection.call_sync(
            destination,
            path,
            interface,
            method,
            GLib.Variant(signature, parameters),
            None,
            Gio.DBusCallFlags.NONE,
            2000,
            None,
        ).unpack()[0]
        if result != expected:
            pending[result] = pending.pop(expected)

    def response(conn, sender, object_path, iface, signal, parameters, user_data):
        callback = pending.pop(object_path, None)
        if callback:
            code, results = parameters.unpack()
            if code != 0:
                fail("GlobalShortcuts setup cancelled or denied")
            else:
                try:
                    callback(results)
                except Exception as exc:
                    fail(str(exc))

    def bound(results):
        state["ready"] = True

    def created(results):
        state["session"] = results["session_handle"]
        token = "mouthkeys_bind_" + uuid.uuid4().hex
        shortcuts = [
            (
                entry["id"],
                {
                    "description": GLib.Variant("s", "MouthKeys " + entry["action"]),
                    "preferred_trigger": GLib.Variant("s", entry["trigger"]),
                },
            )
            for entry in entries
        ]
        request(
            "BindShortcuts",
            "(oa(sa{sv})sa{sv})",
            (state["session"], shortcuts, "", {"handle_token": GLib.Variant("s", token)}),
            token,
            bound,
        )

    def activated(conn, sender, object_path, iface, signal, parameters, user_data):
        session, shortcut, *_ = parameters.unpack()
        if state["ready"] and session == state["session"] and shortcut in by_id:
            print(
                json.dumps(
                    {
                        "action": by_id[shortcut],
                        "id": shortcut,
                        "down": signal == "Activated",
                        "at": time.monotonic(),
                    }
                ),
                flush=True,
            )

    def setup_timeout():
        if not state["ready"]:
            fail("GlobalShortcuts setup timed out")
        return False

    connection.signal_subscribe(
        destination,
        "org.freedesktop.portal.Request",
        "Response",
        None,
        None,
        Gio.DBusSignalFlags.NONE,
        response,
        None,
    )
    for signal in ("Activated", "Deactivated"):
        connection.signal_subscribe(
            destination, interface, signal, path, None, Gio.DBusSignalFlags.NONE, activated, None
        )
    connection.signal_subscribe(
        destination,
        "org.freedesktop.portal.Session",
        "Closed",
        None,
        None,
        Gio.DBusSignalFlags.NONE,
        lambda *_: loop.quit(),
        None,
    )
    token = "mouthkeys_create_" + uuid.uuid4().hex
    request(
        "CreateSession",
        "(a{sv})",
        (
            {
                "handle_token": GLib.Variant("s", token),
                "session_handle_token": GLib.Variant("s", "mouthkeys_" + uuid.uuid4().hex),
            },
        ),
        token,
        created,
    )
    GLib.timeout_add_seconds(120, setup_timeout)
    try:
        loop.run()
    finally:
        for pending_path in list(pending):
            try:
                connection.call_sync(
                    destination,
                    pending_path,
                    "org.freedesktop.portal.Request",
                    "Close",
                    None,
                    None,
                    Gio.DBusCallFlags.NONE,
                    300,
                    None,
                )
            except Exception:
                pass
        if state["session"]:
            connection.call_sync(
                destination,
                state["session"],
                "org.freedesktop.portal.Session",
                "Close",
                None,
                None,
                Gio.DBusCallFlags.NONE,
                500,
                None,
            )


if __name__ == "__main__":
    try:
        helper()
    except Exception as exc:
        print(json.dumps({"error": "GlobalShortcuts portal: " + str(exc)}), flush=True)
        raise SystemExit(1) from None
