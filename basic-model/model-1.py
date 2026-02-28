
# ===============================
# 🔑 SET YOUR NGROK TOKEN HERE
# ===============================
NGROK_AUTH_TOKEN = "YOUR NGROK AUTH TOKEN HERE"

# ===============================
# INSTALL DEPENDENCIES
# ===============================
!pip install -q flask pyngrok transformers torch accelerate

# ===============================
# IMPORTS
# ===============================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify, render_template_string
from pyngrok import ngrok, conf
import threading

# ===============================
# CONFIGURE NGROK
# ===============================
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# ===============================
# LOAD FREE MODEL (TinyLlama)
# ===============================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ===============================
# FLASK APP
# ===============================
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Colab AI Chat</title>
<style>
body{background:#343541;color:white;font-family:Arial;display:flex;justify-content:center;}
.container{width:60%;margin-top:40px;}
.chatbox{background:#444654;padding:20px;border-radius:10px;height:500px;overflow-y:auto;}
.user{color:#4CAF50;margin:10px 0;}
.ai{color:#00BCD4;margin:10px 0;}
input{width:80%;padding:10px;border:none;border-radius:5px;}
button{padding:10px;border:none;border-radius:5px;background:#10A37F;color:white;}
</style>
</head>
<body>
<div class="container">
<h2>🧠 Free Colab AI Agent</h2>
<div class="chatbox" id="chatbox"></div><br>
<input type="text" id="msg" placeholder="Type message...">
<button onclick="send()">Send</button>
</div>
<script>
async function send(){
let input=document.getElementById("msg");
let message=input.value;
if(!message)return;
let box=document.getElementById("chatbox");
box.innerHTML+="<div class='user'><b>You:</b> "+message+"</div>";
input.value="";
let res=await fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:message})});
let data=await res.json();
box.innerHTML+="<div class='ai'><b>AI:</b> "+data.response+"</div>";
box.scrollTop=box.scrollHeight;
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]

    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

# ===============================
# START SERVER + NGROK
# ===============================
def run():
    app.run(port=5000)

threading.Thread(target=run).start()

public_url = ngrok.connect(5000)
print("🌍 Public URL:", public_url)
