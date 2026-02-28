
# ===============================
# 🔑 SET NGROK TOKEN
# ===============================
NGROK_AUTH_TOKEN = "37fJKKexs66q3bWBAAelBYiU2Yp_7Fq6yLN25TUj43fiBHfEN"

# ===============================
# INSTALL
# ===============================
!pip install -q flask pyngrok transformers accelerate bitsandbytes torch

# ===============================
# IMPORTS
# ===============================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from flask import Flask, request, jsonify, render_template_string
from pyngrok import ngrok
import threading

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# ===============================
# MODEL CONFIG
# ===============================
models_dict = {
    "llama3": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "mixtral": "mistralai/Mistral-7B-Instruct-v0.2",
    "deepseek": "deepseek-ai/deepseek-llm-7b-chat"
}

current_model_name = None
model = None
tokenizer = None

# Faster 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

def load_model(model_key):
    global model, tokenizer, current_model_name
    if current_model_name == model_key:
        return
    
    print(f"Loading {model_key}...")
    model_name = models_dict[model_key]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    current_model_name = model_key
    print("Model Loaded!")

# Load default model
load_model("llama3")

# ===============================
# FLASK APP
# ===============================
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Advanced AI Agent</title>
<style>
body{background:#202123;color:white;font-family:Arial;display:flex;justify-content:center;}
.container{width:70%;margin-top:30px;}
.chatbox{background:#343541;padding:20px;border-radius:10px;height:500px;overflow-y:auto;}
.user{color:#4CAF50;margin:10px 0;}
.ai{color:#00BCD4;margin:10px 0;}
select,input{padding:10px;border-radius:5px;border:none;}
button{padding:10px;border:none;border-radius:5px;background:#10A37F;color:white;}
</style>
</head>
<body>
<div class="container">
<h2>🚀 Multi-Model AI Agent</h2>

<select id="modelSelect">
<option value="llama3">Llama 3.1</option>
<option value="mixtral">Mixtral</option>
<option value="deepseek">DeepSeek</option>
</select>

<div class="chatbox" id="chatbox"></div><br>
<input type="text" id="msg" placeholder="Type message...">
<button onclick="send()">Send</button>
</div>

<script>
async function send(){
let input=document.getElementById("msg");
let message=input.value;
let model=document.getElementById("modelSelect").value;
if(!message)return;
let box=document.getElementById("chatbox");
box.innerHTML+="<div class='user'><b>You:</b> "+message+"</div>";
input.value="";
let res=await fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:message,model:model})});
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
    data = request.json
    user_input = data["message"]
    selected_model = data["model"]

    load_model(selected_model)

    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,      # reduced for speed
            temperature=0.6,         # faster stable output
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

# ===============================
# START SERVER
# ===============================
def run():
    app.run(port=5000)

threading.Thread(target=run).start()

public_url = ngrok.connect(5000)
print("🌍 Public URL:", public_url)
