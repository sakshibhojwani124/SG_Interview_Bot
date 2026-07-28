# LocalVoice-AI Setup Guide

## 1. Create Project Folder Structure

```text
project
│
├── SG_AI_Interviewer        (cloned repository)
├── Livekit-Server
└── Llama-Server
```

## 2. Configure Environment Variables

1. Copy the binary file path of the LiveKit server.
2. Add the path to the `LIVEKIT_BIN` environment variable in .env file.
3. Copy the binary file path of the Llama server.
4. Add the path to the `LLAMA_BIN` environment variable in .env file.

---

## 3. Check UV Installation

Verify whether `uv` is installed:

```powershell
uv --version
```

If `uv` is not installed, run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 4. Backend Setup

Open a terminal in the `SG_AI_Interviewer` folder and run:

```powershell
uv python install 3.11
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -e ".[ml,dev]"
python -m livekit.agents download-files
python -m local_voice_ai serve
```

The backend server will start and remain running.

---

## 5. Frontend Setup

Open a **new terminal** in the `SG_AI_Interviewer` folder.

Run:

```powershell
cd frontend
pnpm install
pnpm run dev
```

The frontend development server will start and remain running.
