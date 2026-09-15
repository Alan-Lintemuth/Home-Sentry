from flask import Flask, jsonify, render_template, request
import json
import os
import signal
import subprocess

app = Flask(__name__)
INTERFACE = os.environ.get("HOME_SENTRY_INTERFACE", "wlan0mon")
STATE_FILE = "dashboard_state.json"
SHIELD_PROCESS = None

def reset_state(message="System idle."):
    with open(STATE_FILE, "w") as f:
        json.dump({"status":"Offline","packets":0,"attacks":0,"confidence":0.0,"message":message,"logs":[]}, f)

@app.route("/")
def index():
    reset_state()
    return render_template("dashboard.html")

@app.route("/scan")
def scan_networks():
    command = ["sudo","tshark","-i",INTERFACE,"-a","duration:10","-T","fields","-e","wlan.bssid","-e","wlan.ssid","-Y","wlan.fc.type_subtype == 8"]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        networks, seen = [], set()
        for line in result.stdout.splitlines():
            parts=line.split("\t")
            if len(parts)>=2:
                bssid,ssid=parts[0].strip(),parts[1].strip()
                ssid=ssid.split(",")[0] if ssid else "[Hidden Network]"
                if bssid and bssid not in seen:
                    seen.add(bssid); networks.append({"bssid":bssid,"ssid":ssid})
        return jsonify(networks)
    except Exception as exc:
        return jsonify({"error":str(exc)}),500

@app.route("/start", methods=["POST"])
def start_capture():
    global SHIELD_PROCESS
    bssid=(request.get_json(silent=True) or {}).get("bssid")
    if not bssid: return jsonify({"error":"No target selected"}),400
    if SHIELD_PROCESS and SHIELD_PROCESS.poll() is None: return jsonify({"error":"Shield is already running"}),409
    reset_state("Starting live analysis...")
    SHIELD_PROCESS=subprocess.Popen(["python","-m","home_sentry.live_detector","--interface",INTERFACE,"--target",bssid])
    return jsonify({"status":"Started","pid":SHIELD_PROCESS.pid})

@app.route("/stop", methods=["POST"])
def stop_capture():
    global SHIELD_PROCESS
    if SHIELD_PROCESS and SHIELD_PROCESS.poll() is None:
        SHIELD_PROCESS.send_signal(signal.SIGINT); SHIELD_PROCESS.wait(timeout=15)
    SHIELD_PROCESS=None
    reset_state("Shield deactivated. Forensic logs saved.")
    return jsonify({"status":"stopped"})

@app.route("/status")
def get_status():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f: return jsonify(json.load(f))
        except (OSError, json.JSONDecodeError): pass
    return jsonify({"status":"Offline","packets":0,"attacks":0,"logs":[]})

@app.route("/shutdown", methods=["POST"])
def shutdown():
    stop_capture()
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"status":"shutting down"})

if __name__ == "__main__":
    reset_state()
    app.run(host="127.0.0.1", port=5000, debug=False)
