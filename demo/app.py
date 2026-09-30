import json
import math
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime

from flask import Flask, jsonify, render_template_string
import paho.mqtt.client as mqtt

DB_PATH = "motor.db"
MQTT_HOST = "localhost"
MQTT_PORT = 1883
TOPIC = "factory/motor1/telemetry"
FAULT_TOPIC = "factory/motor1/alert"
WINDOW_SIZE = 8

app = Flask(__name__)
state_lock = threading.Lock()
vibration_window = deque(maxlen=WINDOW_SIZE)

latest = {
    "timestamp": None,
    "temperature": None,
    "vibration": None,
    "rms": None,
    "status": "WAITING",
    "fault_probability": 0.0,
    "mqtt_connected": False,
    "window_size": WINDOW_SIZE,
}

HTML = r'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Motor Edge Monitoring</title>
  <style>
    :root {
      --bg:#f4f6f8; --panel:#ffffff; --text:#17212b; --muted:#67727e;
      --border:#d9e0e6; --ok:#2e7d32; --warn:#b26a00; --bad:#c62828; --blue:#1565c0;
    }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Arial,Helvetica,sans-serif; background:var(--bg); color:var(--text); }
    .wrap { max-width:1180px; margin:0 auto; padding:22px; }
    .top { display:flex; justify-content:space-between; gap:14px; align-items:center; flex-wrap:wrap; }
    h1 { margin:0 0 4px; font-size:30px; }
    .subtitle { color:var(--muted); margin-bottom:18px; }
    .badge { display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:999px; background:#fff; border:1px solid var(--border); font-weight:700; }
    .dot { width:10px; height:10px; border-radius:50%; background:#9aa3aa; }
    .grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }
    .card { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:18px; box-shadow:0 2px 8px rgba(20,30,40,.05); }
    .metric-label { font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
    .metric { font-size:32px; font-weight:800; margin-top:8px; }
    .unit { font-size:16px; color:var(--muted); font-weight:600; }
    .status-card { margin-top:14px; display:grid; grid-template-columns:1.2fr 1fr; gap:14px; }
    .status-box { border-radius:14px; padding:22px; background:#eef1f4; border:1px solid var(--border); }
    .status-big { font-size:42px; font-weight:900; margin:6px 0; }
    .alert { display:none; margin-top:12px; padding:12px 14px; border-radius:10px; font-weight:800; background:#ffebee; color:#9b1c1c; border:1px solid #ffcdd2; }
    .gauge { margin-top:14px; }
    .track { height:18px; background:#e8edf1; border-radius:999px; overflow:hidden; border:1px solid #d7dde2; }
    .fill { height:100%; width:0%; background:var(--blue); transition:width .35s ease; }
    .scale { display:flex; justify-content:space-between; font-size:12px; color:var(--muted); margin-top:5px; }
    .charts { margin-top:14px; display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    canvas { width:100%; height:250px; display:block; background:#fff; border-radius:10px; }
    .chart-title { font-weight:800; margin-bottom:8px; }
    .legend { color:var(--muted); font-size:12px; margin-top:5px; }
    .table-card { margin-top:14px; overflow:auto; }
    table { width:100%; border-collapse:collapse; font-size:14px; }
    th,td { padding:9px 10px; border-bottom:1px solid #e6eaee; text-align:left; white-space:nowrap; }
    th { background:#fafbfc; position:sticky; top:0; }
    .footer-note { color:var(--muted); margin-top:14px; font-size:13px; }
    code { background:#eef1f4; padding:2px 5px; border-radius:4px; }
    @media (max-width:900px){ .grid{grid-template-columns:repeat(2,1fr)} .status-card,.charts{grid-template-columns:1fr} }
    @media (max-width:520px){ .grid{grid-template-columns:1fr} .wrap{padding:12px} }
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <h1>Smart Motor Edge Monitoring</h1>
      <div class="subtitle">MQTT telemetry → edge analytics → SQLite → REST dashboard</div>
    </div>
    <div class="badge"><span class="dot" id="mqttDot"></span><span id="mqttText">MQTT: checking…</span></div>
  </div>

  <div class="grid">
    <div class="card"><div class="metric-label">Temperature</div><div class="metric"><span id="temp">--</span> <span class="unit">°C</span></div></div>
    <div class="card"><div class="metric-label">Current vibration</div><div class="metric"><span id="vib">--</span> <span class="unit">g</span></div></div>
    <div class="card"><div class="metric-label">Window RMS</div><div class="metric"><span id="rms">--</span> <span class="unit">g</span></div></div>
    <div class="card"><div class="metric-label">Fault probability</div><div class="metric"><span id="prob">--</span> <span class="unit">%</span></div></div>
  </div>

  <div class="status-card">
    <div class="status-box" id="statusBox">
      <div class="metric-label">Machine status</div>
      <div class="status-big" id="status">WAITING</div>
      <div id="time" class="subtitle" style="margin:0"></div>
      <div class="alert" id="alertBox">⚠ High vibration detected — inspect Motor 1.</div>
    </div>
    <div class="card">
      <div class="metric-label">Fault-risk gauge</div>
      <div class="gauge">
        <div class="track"><div class="fill" id="riskFill"></div></div>
        <div class="scale"><span>0%</span><span>50%</span><span>100%</span></div>
      </div>
      <p class="footer-note">RMS uses the latest <b id="winSize">--</b> vibration readings. Thresholds: NORMAL &lt; 0.8, WARNING 0.8–1.5, FAULT ≥ 1.5.</p>
    </div>
  </div>

  <div class="charts">
    <div class="card">
      <div class="chart-title">Live Vibration and RMS</div>
      <canvas id="vibChart" width="560" height="250"></canvas>
      <div class="legend">Vibration = solid line · RMS = dashed line · threshold guides at 0.8 and 1.5 g</div>
    </div>
    <div class="card">
      <div class="chart-title">Live Temperature</div>
      <canvas id="tempChart" width="560" height="250"></canvas>
      <div class="legend">Latest values are read from the local SQLite history through the REST API.</div>
    </div>
  </div>

  <div class="card table-card">
    <div class="chart-title">Recent edge readings</div>
    <table>
      <thead><tr><th>Time</th><th>Temperature</th><th>Vibration</th><th>RMS</th><th>Risk</th><th>Status</th></tr></thead>
      <tbody id="historyBody"><tr><td colspan="6">Waiting for telemetry…</td></tr></tbody>
    </table>
  </div>

  <div class="footer-note">
    REST endpoints: <code>/api/status</code> and <code>/api/history</code>. The browser is only a viewer; the core fault decision happens locally in <code>app.py</code>.
  </div>
</div>
<script>
function statusStyle(status){
  if(status==='NORMAL') return {bg:'#e8f5e9', fg:'#1b5e20', bar:'#2e7d32'};
  if(status==='WARNING') return {bg:'#fff3e0', fg:'#8a4b00', bar:'#ef8f00'};
  if(status==='FAULT') return {bg:'#ffebee', fg:'#a11d1d', bar:'#c62828'};
  return {bg:'#eef1f4', fg:'#4c5964', bar:'#1565c0'};
}

function drawLineChart(canvasId, series, opts={}){
  const c=document.getElementById(canvasId), ctx=c.getContext('2d');
  const W=c.width, H=c.height, pad={l:42,r:12,t:12,b:28};
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);
  const all=[]; series.forEach(s=>s.values.forEach(v=>{if(v!=null) all.push(v)}));
  if(opts.guides) opts.guides.forEach(g=>all.push(g));
  if(!all.length){ctx.fillStyle='#7b8790';ctx.font='14px Arial';ctx.fillText('Waiting for data…',18,30);return;}
  let ymin=opts.ymin!=null?opts.ymin:Math.min(...all), ymax=opts.ymax!=null?opts.ymax:Math.max(...all);
  if(ymax===ymin) ymax=ymin+1;
  const margin=(ymax-ymin)*0.12; ymin=Math.max(0,ymin-margin); ymax+=margin;
  const x=(i,n)=>pad.l+(n<=1?0:i*(W-pad.l-pad.r)/(n-1));
  const y=v=>pad.t+(ymax-v)*(H-pad.t-pad.b)/(ymax-ymin);
  ctx.strokeStyle='#e3e8ec';ctx.lineWidth=1;
  for(let k=0;k<=4;k++){const yy=pad.t+k*(H-pad.t-pad.b)/4;ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(W-pad.r,yy);ctx.stroke(); const val=ymax-k*(ymax-ymin)/4;ctx.fillStyle='#6c7882';ctx.font='11px Arial';ctx.fillText(val.toFixed(1),4,yy+4);}
  if(opts.guides){ctx.save();ctx.setLineDash([5,5]);ctx.strokeStyle='#9aa3aa';opts.guides.forEach(g=>{ctx.beginPath();ctx.moveTo(pad.l,y(g));ctx.lineTo(W-pad.r,y(g));ctx.stroke();});ctx.restore();}
  series.forEach((s,si)=>{const n=s.values.length;ctx.save();ctx.strokeStyle=si===0?'#1565c0':'#7b1fa2';ctx.lineWidth=2.3;if(s.dashed)ctx.setLineDash([7,5]);ctx.beginPath();let started=false;s.values.forEach((v,i)=>{if(v==null)return;const xx=x(i,n),yy=y(v);if(!started){ctx.moveTo(xx,yy);started=true}else ctx.lineTo(xx,yy)});ctx.stroke();ctx.restore();});
  ctx.strokeStyle='#6b747c';ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,H-pad.b);ctx.lineTo(W-pad.r,H-pad.b);ctx.stroke();
  ctx.fillStyle='#6c7882';ctx.font='11px Arial';ctx.fillText('older',pad.l,H-8);ctx.fillText('latest',W-pad.r-30,H-8);
}

async function load(){
  try{
    const [sr,hr]=await Promise.all([fetch('/api/status'),fetch('/api/history')]);
    const d=await sr.json(), hist=await hr.json();
    document.getElementById('temp').textContent=d.temperature==null?'--':d.temperature.toFixed(2);
    document.getElementById('vib').textContent=d.vibration==null?'--':d.vibration.toFixed(2);
    document.getElementById('rms').textContent=d.rms==null?'--':d.rms.toFixed(2);
    document.getElementById('prob').textContent=(100*(d.fault_probability||0)).toFixed(0);
    document.getElementById('status').textContent=d.status;
    document.getElementById('time').textContent=d.timestamp?('Last telemetry: '+d.timestamp):'Waiting for telemetry…';
    document.getElementById('winSize').textContent=d.window_size||'--';
    const st=statusStyle(d.status), box=document.getElementById('statusBox'); box.style.background=st.bg; box.style.color=st.fg;
    document.getElementById('riskFill').style.width=(100*(d.fault_probability||0))+'%'; document.getElementById('riskFill').style.background=st.bar;
    document.getElementById('alertBox').style.display=d.status==='FAULT'?'block':'none';
    const dot=document.getElementById('mqttDot'), mt=document.getElementById('mqttText');
    dot.style.background=d.mqtt_connected?'#2e7d32':'#c62828'; mt.textContent=d.mqtt_connected?'MQTT: connected':'MQTT: disconnected';

    const rows=[...hist].reverse();
    drawLineChart('vibChart',[{values:rows.map(r=>r.vibration)},{values:rows.map(r=>r.rms),dashed:true}],{guides:[0.8,1.5],ymin:0});
    drawLineChart('tempChart',[{values:rows.map(r=>r.temperature)}],{ymin:0});
    const body=document.getElementById('historyBody');
    body.innerHTML=hist.length?hist.slice(0,12).map(r=>`<tr><td>${r.timestamp||''}</td><td>${r.temperature.toFixed(2)} °C</td><td>${r.vibration.toFixed(2)} g</td><td>${r.rms.toFixed(2)} g</td><td>${(100*r.fault_probability).toFixed(0)}%</td><td><b>${r.status}</b></td></tr>`).join(''):'<tr><td colspan="6">Waiting for telemetry…</td></tr>';
  }catch(e){document.getElementById('mqttText').textContent='Dashboard API unavailable';}
}
setInterval(load,1000); load();
</script>
</body>
</html>
'''


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS readings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            temperature REAL,
            vibration REAL,
            rms REAL,
            fault_probability REAL,
            status TEXT
        )
    """)
    con.commit()
    con.close()


def edge_analytics(vibration: float):
    """Calculate true RMS over a rolling window of recent vibration readings."""
    vibration_window.append(vibration)
    rms = math.sqrt(sum(x * x for x in vibration_window) / len(vibration_window))

    if rms < 0.8:
        prob = 0.05
        status = "NORMAL"
    elif rms < 1.5:
        prob = 0.45
        status = "WARNING"
    else:
        prob = 0.92
        status = "FAULT"
    return rms, prob, status


def on_connect(client, userdata, flags, reason_code, properties=None):
    connected = int(reason_code) == 0
    with state_lock:
        latest["mqtt_connected"] = connected
    print(f"[EDGE] Connected to MQTT broker: {reason_code}")
    if connected:
        client.subscribe(TOPIC, qos=1)
        print(f"[EDGE] Subscribed to {TOPIC}")


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    with state_lock:
        latest["mqtt_connected"] = False
    print(f"[EDGE] MQTT disconnected: {reason_code}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        temperature = float(payload["temperature"])
        vibration = float(payload["vibration"])
        ts = payload.get("timestamp") or datetime.now().isoformat(timespec="seconds")

        rms, prob, status = edge_analytics(vibration)

        with state_lock:
            latest.update({
                "timestamp": ts,
                "temperature": temperature,
                "vibration": vibration,
                "rms": rms,
                "status": status,
                "fault_probability": prob,
            })

        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO readings(timestamp, temperature, vibration, rms, fault_probability, status) VALUES(?,?,?,?,?,?)",
            (ts, temperature, vibration, rms, prob, status),
        )
        con.commit()
        con.close()

        print(f"[EDGE] T={temperature:.2f} C  V={vibration:.2f} g  RMS={rms:.2f}  STATUS={status}")

        if status == "FAULT":
            alert = {
                "timestamp": ts,
                "machine": "motor1",
                "status": status,
                "fault_probability": prob,
                "rms": rms,
                "message": "Possible bearing fault: inspect motor",
            }
            client.publish(FAULT_TOPIC, json.dumps(alert), qos=1)
            print("[EDGE] Published fault alert")
    except Exception as e:
        print("[EDGE] Bad message:", e)


def mqtt_worker():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edge-processor")
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    while True:
        try:
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            client.loop_forever()
        except Exception as e:
            with state_lock:
                latest["mqtt_connected"] = False
            print("[EDGE] MQTT unavailable, retrying in 2 s:", e)
            time.sleep(2)


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/status")
def api_status():
    with state_lock:
        snapshot = dict(latest)
    return jsonify(snapshot)


@app.route("/api/history")
def api_history():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT timestamp, temperature, vibration, rms, fault_probability, status FROM readings ORDER BY id DESC LIMIT 60"
    ).fetchall()
    con.close()
    return jsonify([
        {
            "timestamp": r[0],
            "temperature": r[1],
            "vibration": r[2],
            "rms": r[3],
            "fault_probability": r[4],
            "status": r[5],
        }
        for r in rows
    ])


if __name__ == "__main__":
    init_db()
    threading.Thread(target=mqtt_worker, daemon=True).start()
    print("[WEB] Dashboard: http://127.0.0.1:5000")
    print("[WEB] REST:      http://127.0.0.1:5000/api/status")
    print("[WEB] History:   http://127.0.0.1:5000/api/history")
    app.run(host="127.0.0.1", port=5000, debug=False)
