#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "litellm",
#     "flask",
#     "markdown-it-py[linkify,plugins]",
# ]
# ///

from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from html import escape
from pathlib import Path
from typing import Dict, List
from uuid import uuid4
from flask import Flask, Response, g, render_template_string, request, session
from litellm import completion
from markdown_it import MarkdownIt

CHAT_DIR = Path(__file__).with_suffix("").parent / "chats"
CHAT_DIR.mkdir(exist_ok=True)
HTMX = "https://unpkg.com/htmx.org@2.0.4"
SSE  = "https://unpkg.com/htmx-ext-sse@2.2.3"

md = (
    MarkdownIt("commonmark", {"linkify": True})
    .enable("table")
    .enable("strikethrough")
)

DEFAULT_LLM_CONFIG: Dict[str, str] = {
    "model": "openai/local",
    "api_base": "http://localhost:1234/v1",  # LM Studio default
    "api_key": "dummy",
}

app = Flask(__name__)
app.secret_key = "dev-secret-change-me" 


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    message: str


@dataclass
class Conversation:
    sid: str
    messages: List[Message] = field(default_factory=list)

    @classmethod
    def current(cls) -> "Conversation":
        """Return active conversation for this session (cached on g)."""
        if hasattr(g, "conv"):
            return g.conv

        sid = session.setdefault("sid", "conv-" + uuid4().hex)
        path = CHAT_DIR / f"{sid}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            conv = cls(sid=sid, messages=[Message(**d) for d in data])
        else:
            conv = cls(
                sid=sid,
                messages=[Message(role="system", message="You are a helpful assistant.")],
            )
        g.conv = conv
        return conv

    def save(self) -> None:
        path = CHAT_DIR / f"{self.sid}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([asdict(m) for m in self.messages], fh, ensure_ascii=False, indent=2)

    def to_html(self) -> str:
        return "".join(render_message_html(m) for m in self.messages if m.role != "system")

def render_message_html(msg: Message) -> str:
    cls = "user" if msg.role == "user" else "bot"
    body = escape(msg.message) if msg.role == "user" else md.render(msg.message)
    return f'<div class="message {cls}">{body}</div>'


def _sse_chunk(html: str) -> str:
    """Wrap HTML lines as SSE data: lines must start with `data: `."""
    return "\n".join(f"data: {line}" for line in html.splitlines()) + "\n\n"


def get_llm_config() -> Dict[str, str]:
    """Return provider settings stored in session (or defaults)."""
    return session.get("llm_config", DEFAULT_LLM_CONFIG)


TEMPLATE = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Chat</title>
  <script src=\"{HTMX}\"></script>
  <script src=\"{SSE}\"></script>
  <style>
    :root {{ --bg:#202124; --panel:#303134; --text:#e8eaed; --accent:#8ab4f8; --border:#5f6368; }}
    html,body {{ height:100%; margin:0; }}
    body {{ font-family:Arial,sans-serif; background:var(--bg); color:var(--text); display:flex; }}
    #sidebar {{ width:220px; background:#111; border-right:1px solid var(--border); overflow-y:auto; padding:.5rem; }}
    #chatbox {{ flex:1; display:flex; flex-direction:column; }}
    #messages {{ flex:1; overflow-y:auto; padding:1rem; }}
    .message {{ margin:.5rem 0; padding:.5rem .75rem; border-radius:12px; max-width:80%; background:var(--panel); }}
    .user {{ background:#e0e0e0; color:var(--bg); margin-left:auto; }}
    .bot  {{ background:#3c4043; color:var(--text); margin-right:auto; }}
    form {{ display:flex; gap:.5rem; padding:.75rem; border-top:1px solid var(--border); background:var(--bg); }}
    input {{ flex:1; padding:.5rem; border:1px solid var(--border); border-radius:4px; background:var(--panel); color:var(--text); }}
    button {{ padding:.5rem 1rem; border:none; background:var(--accent); color:var(--bg); border-radius:4px; }}
    .chat-link, .action-btn {{ display:block; padding:.4rem .6rem; color:#ccc; border-radius:4px; text-align:left; width:100%; background:none; border:none; }}
    .action-btn:hover, .chat-link:hover {{ background:#222; cursor:pointer; }}
  </style>
</head>
<body hx-ext=\"sse\">
  <aside id=\"sidebar\" hx-get=\"/sidebar\" hx-trigger=\"load, every 30s\" hx-swap=\"innerHTML\"></aside>
  <div id=\"chatbox\">
    <div id=\"messages\"></div>
    <form hx-post=\"/input\" hx-target=\"#messages\" hx-swap=\"beforeend\" hx-trigger=\"submit\">
      <input name=\"prompt\" placeholder=\"Say something…\" autocomplete=\"off\" required>
      <button>Send</button>
    </form>
  </div>
</body>
</html>"""

@app.route("/")
def index():
    conv = Conversation.current()
    return render_template_string(TEMPLATE, messages=conv.to_html())

@app.route("/input", methods=["POST"])
def input_route():
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return ""

    conv = Conversation.current()

    # Store title for sidebar (first user msg)
    if len(conv.messages) == 1:
        session["title"] = prompt[:40]

    conv.messages.append(Message(role="user", message=prompt))
    conv.save()

    user_html = render_message_html(conv.messages[-1])

    placeholder = (
        '<div class="message bot" '
        'sse-connect="/stream" sse-swap="message" sse-close="done" '
        'hx-swap="innerHTML"></div>'
    )
    return user_html + placeholder


@app.route("/stream")
def stream():
    conv = Conversation.current()
    cfg = get_llm_config()

    def gen():
        buffer = ""
        for chunk in completion(
            model=cfg["model"],
            api_base=cfg["api_base"],
            api_key=cfg.get("api_key"),
            messages=[{"role": m.role, "content": m.message} for m in conv.messages],
            stream=True,
        ):
            token = chunk.choices[0].delta.get("content", "")
            if token:
                buffer += token
                yield _sse_chunk(md.render(buffer))

        conv.messages.append(Message(role="assistant", message=buffer))
        conv.save()
        yield "event: done\ndata:\n\n"

    return Response(
        gen(),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        mimetype="text/event-stream",
    )

@app.route("/sidebar")
def sidebar():
    # Action buttons
    buttons = """
    <button class=\"action-btn\" hx-get=\"/new\" hx-target=\"#messages\" hx-swap=\"innerHTML\">➕ New Chat</button>
    <button class=\"action-btn\" hx-get=\"/settings\" hx-target=\"#messages\" hx-swap=\"innerHTML\">⚙️ Settings</button>
    <hr style=\"border:1px solid var(--border);\">
    """

    # Existing chats
    links: List[str] = []
    for path in sorted(CHAT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        sid = path.stem
        title = "Untitled chat"
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                for d in data:
                    if d["role"] == "user":
                        title = d["message"][:40] + ("…" if len(d["message"]) > 40 else "")
                        break
        except Exception:
            pass
        active = " style=\"background:#333;\"" if sid == session.get("sid") else ""
        links.append(
            f'<a class="chat-link" hx-get="/load?sid={sid}" hx-target="#messages" '
            f'hx-swap="innerHTML"{active}>{escape(title)}</a>'
        )

    return buttons + "<h2>Your chats</h2>" + "".join(links)


@app.route("/load")
def load():
    sid = request.args.get("sid")
    if not sid:
        return ""
    session["sid"] = sid
    conv = Conversation.current()
    return conv.to_html()


@app.route("/new")
def new_chat():
    """Start a fresh conversation (keeps provider settings)."""
    session["sid"] = "conv-" + uuid4().hex
    conv = Conversation.current()
    conv.save()
    return conv.to_html()

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        # Persist new provider settings
        session["llm_config"] = {
            "model": request.form.get("model", DEFAULT_LLM_CONFIG["model"]).strip() or DEFAULT_LLM_CONFIG["model"],
            "api_base": request.form.get("api_base", DEFAULT_LLM_CONFIG["api_base"]).strip() or DEFAULT_LLM_CONFIG["api_base"],
            "api_key": request.form.get("api_key", DEFAULT_LLM_CONFIG["api_key"]).strip() or DEFAULT_LLM_CONFIG["api_key"],
        }
        return (
            '<div style="padding:1rem;"><h2>Settings saved</h2>'
            '<p>Your LLM provider configuration has been updated. 🎉</p></div>'
        )

    # GET display settings form
    cfg = get_llm_config()
    form = f"""
    <div style=\"padding:1rem;\">
      <h2>Settings</h2>
      <form hx-post=\"/settings\" hx-target=\"#messages\" hx-swap=\"innerHTML\" style=\"display:flex; flex-direction:column; gap:.5rem;\">
        <label>Model <input name=\"model\" value=\"{escape(cfg['model'])}\"></label>
        <label>API Base <input name=\"api_base\" value=\"{escape(cfg['api_base'])}\"></label>
        <label>API Key <input name=\"api_key\" value=\"{escape(cfg['api_key'])}\"></label>
        <button style=\"margin-top:.5rem;\">Save</button>
      </form>
      <p style=\"font-size:.9rem;opacity:.8;\"><strong>Examples</strong><br>
        Ollama server: <code>http://localhost:11434</code><br>
        LM Studio: <code>http://localhost:1234/v1</code><br>
        llama.cpp server: <code>http://localhost:8080/v1</code><br>
        llamafile: <code>http://localhost:4891/v1</code>
      </p>
    </div>
    """
    return form

def main():
    app.run(port=8000, debug=True)

if __name__ == "__main__":
    main()
