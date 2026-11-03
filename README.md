<div align="center">

# 🤖 End-to-End Agentic AI & Automation Lab

**A comprehensive, production-grade repository for building, deploying, and managing intelligent AI agents, RAG pipelines, and automated workflows.**

[![GitHub stars](https://img.shields.io/github/stars/MDalamin5/End-to-End-Agentic-Ai-Automation-Lab?style=social)](https://github.com/MDalamin5/End-to-End-Agentic-Ai-Automation-Lab/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MDalamin5/End-to-End-Agentic-Ai-Automation-Lab?style=social)](https://github.com/MDalamin5/End-to-End-Agentic-Ai-Automation-Lab/network/members)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

[Overview](#-overview) •
[Key Highlights](#-key-highlights) •
[Agent Harness](#-what-is-an-agent-harness) •
[Project Architecture](#-repository-modules--projects) •
[Tech Stack](#️-tech-stack--tools) •
[Getting Started](#-getting-started)

</div>

---

## 📖 Overview

Welcome to the **End-to-End Agentic AI Automation Lab**. This repository is a massive, hands-on engineering playbook demonstrating how to transition from basic LLM API calls to **complex, multi-agent autonomous systems** and **production-ready AI products**. 

Whether you are looking to build highly reliable Agentic workflows using **LangGraph**, orchestrate multi-agent collaboration via **AutoGen**, implement cutting-edge **Model Context Protocol (MCP)**, or serve fine-tuned local models using **vLLM** and **Unsloth**, this repository has you covered.

---

## 🚀 Key Highlights

* **Advanced Agentic Frameworks**: Deep dives into **LangGraph** (StateGraphs, subgraphs, memory, HITL) and **AutoGen** (RoundRobin, Swarm, custom tools).
* **Model Context Protocol (MCP)**: Industry-grade implementations of Anthropic's MCP for tool execution, web search, and Notion integration.
* **Production RAG Systems**: Implementation of Hybrid Search, BM25, LlamaParse, Semantic Routing, and Long/Short-Term Memory (Mem0).
* **AI Workflow Automation**: Zero-code/low-code multi-agent orchestration using **n8n** and **LangFlow**.
* **LLM Fine-Tuning & Serving**: Hands-on pipelines for fine-tuning with **LoRA/Unsloth** and deploying high-throughput inference endpoints with **vLLM**.
* **End-to-End Products**: Complete full-stack implementations of an AI Interviewer, a Production ATS, and SynapseAI (a stateful, persistent chatbot).

---

## 🧰 What is an Agent Harness?

An LLM on its own can only take text in and send text out. The **agent harness** is everything you build *around* the model so it can work in the real world and finish multi-step tasks. A simple way to put it: **Agent = Model + Harness**.

### What's in a harness

| Component | Role |
|---|---|
| **System prompt** | Sets the agent's role, rules, and how it should behave |
| **Tools** | Functions the model can call: search, SQL, code execution, APIs, MCP servers |
| **Context management** | Picks what goes into the context window: summarization, context offloading, prompt caching |
| **Memory** | Short-term (the conversation/state) and long-term (facts kept across sessions) |
| **Filesystem / sandbox** | A place to write notes, save large results, and run code safely |
| **Planning** | A to-do list that breaks a big task into steps |
| **Subagents** | Specialized helper agents the main agent hands work to, each with its own clean context |
| **Skills** | Reusable instructions and resources loaded only when they're needed |
| **Human-in-the-loop (HITL)** | Pauses so a human can approve, edit, or reject sensitive actions |
| **Middleware / hooks** | Code that runs before or after the model call to change inputs, outputs, or tool calls |

**The job of a harness:** give the model the *right context at the right time* for the task in front of it.

### Why you need one
* Lets the agent **take actions** in an environment, not just answer questions.
* Connects the agent to **your data** and systems.
* Keeps a **growing context** under control over long tasks.
* Supports **HITL** approval for risky steps.
* Runs work **in parallel** through subagents.
* Helps the agent **improve over time** through memory and feedback.

### How the agent loop works

```
            ┌──────────────────────────────────────────┐
 User ───►  │  Harness builds context                  │
 task       │  (system prompt + memory + history +     │
            │   available tools/skills)                │
            └───────────────┬──────────────────────────┘
                            ▼
                  ┌───────────────────┐
                  │   LLM (reasoning) │◄──────────────┐
                  └─────────┬─────────┘               │
             final answer?  │  tool call?             │
              ┌─────────────┴─────────────┐           │
              ▼                           ▼           │
        Return to user        Harness runs the tool  │
                              (middleware, HITL      │
                               checks, sandbox)      │
                                          │          │
                                          └─ result ─┘
                                     added back to context
```

1. **Input** – The user sends a task. The harness builds the prompt from the system prompt, memory, conversation history, and the tool definitions.
2. **Reason** – The model decides what to do: answer right away, call a tool, write a plan, or hand work to a subagent.
3. **Act** – The harness runs the tool call. Middleware can check or change it first, and HITL can pause for approval.
4. **Observe** – The tool result goes back into the context. Large outputs can be saved to the filesystem so the context stays small.
5. **Repeat** – The loop keeps going until the model gives a final answer or hits a stop condition.

### Working with an agent harness (Deep Agents example)

[`deepagents`](https://github.com/langchain-ai/deepagents) is a ready-made harness built on LangGraph. It already includes planning, a virtual filesystem, subagents, and summarization, so you only add your model, prompt, and tools:

```python
from deepagents import create_deep_agent

def get_sum(a: int, b: int) -> int:
    """Return the sum of two numbers."""
    return a + b

agent = create_deep_agent(
    model=model,                              # any LangChain chat model
    system_prompt="You are a helpful math assistant.",
    tools=[get_sum],                          # your custom tools
    # subagents=[...], middleware=[...], interrupt_on={...}  # optional
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the sum of 5 and 9?"}]}
)
print(result["messages"][-1].content)
```

**Tips for working with agents**
* **Write a clear system prompt.** Spell out the goal, the rules, and what output you expect.
* **Keep tools small and well documented.** The model picks tools based on their names, docstrings, and type hints.
* **Give the agent only the tools it needs.** Too many tools make it worse at choosing.
* **Use subagents for big or separate subtasks** so the main agent's context stays clean.
* **Add HITL for irreversible actions** such as sending emails, writing to a database, or making payments.
* **Watch and trace runs** (for example with LangSmith) to debug the agent's decisions.

👉 Hands-on code: [`28-Intorduction-to-deep-and-Agents-Harness`](28-Intorduction-to-deep-and-Agents-Harness/)

---

## 📂 Repository Modules & Projects

The lab is structured progressively. Click to expand each module to see the underlying projects:

<details>
<summary><b>1️⃣ Foundations & Data Ingestion (Modules 01 - 02)</b></summary>
<br>

* `01-Pydantic-Data-Validation`: Data structuring, field validation, and structured LLM outputs.
* `02-LangChain-Basics`: Embedding models, VectorDBs (FAISS, Pinecone), and basic Retrieval-Augmented Generation (RAG) scratchpads.
</details>

<details>
<summary><b>2️⃣ LangGraph & Workflow Orchestration (Modules 03 - 04, 13 - 14)</b></summary>
<br>

* `03-LangGraph-Introduction`: StateGraphs, Agentic workstations, multi-tool calling.
* `04-LangGraph-Agentic-Workflows`: Agentic RAG, Multi-Agent Supervisors, Human-in-the-Loop (HITL), and Corrective RAG (CRAG).
* `13-e2e-Deep-Agents`: Observation, evaluation, and reliable LangGraph applications.
* `14-e2e-Ambient-Agent`: Building background-running autonomous agents.
</details>

<details>
<summary><b>3️⃣ AutoGen Multi-Agent Systems (Modules 05 - 09)</b></summary>
<br>

* `05-Autogen-Introduction`: Async capabilities, tools, and basic teams.
* `06-Autogen-HITL-and-Agentic-Orchestrator`: Selector Group Chats, Docker code execution, and Graph-based AutoGen.
* `07-End-To-End-Projects-Autogen`: GPT Analyzer (Modular architecture), AI Interviewer.
* `08-Advanced-Autogen-Team`: Swarm logic and Society of Mind teams.
* `09-Autogen-RAG-and-Memory`: Integrating `mem0` for cross-session AutoGen memory.
</details>

<details>
<summary><b>4️⃣ Model Context Protocol (MCP) & n8n (Modules 10 - 12)</b></summary>
<br>

* `10-MCP-All-You-Need`: Bridging AutoGen and LangChain with MCP. Lead collector, FireCrawl MCP, and Playwright MCP.
* `11-MCP-based-End-to-End-Products`: Building fast, robust API backends utilizing MCP architectures via ngrok and FastAPI.
* `12-n8n`: High-level automations. Chain of Agents, Social Media Content Generation, parallel agent logic, and Telegram bot integrations.
</details>

<details>
<summary><b>5️⃣ Production RAG & Guardrails (Modules 17, 19)</b></summary>
<br>

* `17-Guardrails-for-llm`: Implementing NeMo Guardrails for secure and constrained LLM outputs.
* `19-Productions-RAG`: Industry-practice RAG including `LlamaParse`, BM25/Hybrid Search, HyDE, chunking strategies, and Reranking pipelines.
</details>

<details>
<summary><b>6️⃣ LLM Fine-Tuning & Deployment (Modules 21 - 22)</b></summary>
<br>

* `21-LLM-Deployment-vLLM`: Deploying models for high-throughput generation using vLLM and accessing via LangChain SDK.
* `22-LLM-FineTune-Deployment`: Model fine-tuning using **Unsloth**, **LoRA**, HuggingFace Pipelines, and quantization setups for edge devices.
</details>

<details>
<summary><b>7️⃣ End-to-End Full-Stack Projects (Modules 18, 20, 23, 24)</b></summary>
<br>

* `18-e2e-chatbot-mem0-tools-HITL-MCP-RAG`: A massive implementation of a fully-featured chatbot with long/short-term memory, PostgreSQL persistence, and streaming UI.
* `20-e2e-Productions-grade-ATS`: End-to-end Applicant Tracking System backed by Alembic, SQLModel, and LangGraph.
* `23-e2e-multi-agent-plan-research-write-blog`: A multi-agent writer architecture with a beautiful web frontend.
* `24-SynapseAI-parsitence-chatbot`: A modern API-first chatbot backend via FastAPI with complex graph routing.
</details>

<details>
<summary><b>8️⃣ Deep Agents & Agent Harness (Module 28)</b></summary>
<br>

* `28-Intorduction-to-deep-and-Agents-Harness`: What an agent harness is, Deep Agents architecture (execution environment, context management, delegation, steering), middleware, system prompts, prompt caching, and search/SQL agents built with `deepagents`.
</details>

---

## 🛠️ Tech Stack & Tools

**Core AI/ML:**
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![LangChain](https://img.shields.io/badge/🦜_LangChain-02569B?style=for-the-badge&logoColor=white)
![vLLM](https://img.shields.io/badge/vLLM-000000?style=for-the-badge&logo=v&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-FFD21E?style=for-the-badge&logoColor=black)

**Agentic & Orchestration:**
![LangGraph](https://img.shields.io/badge/🕸️_LangGraph-1C3C3C?style=for-the-badge&logoColor=white)
![AutoGen](https://img.shields.io/badge/🤖_AutoGen-4B0082?style=for-the-badge&logoColor=white)
![n8n](https://img.shields.io/badge/n8n-%23EA4B3D.svg?style=for-the-badge&logo=n8n&logoColor=white)
![MCP](https://img.shields.io/badge/MCP_(Anthropic)-5B0000?style=for-the-badge)

**Backend & Data:**
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

---

## ⚙️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/MDalamin5/End-to-End-Agentic-Ai-Automation-Lab.git
cd End-to-End-Agentic-Ai-Automation-Lab
```

### 2. Set Up Virtual Environment
It is recommended to use `conda` or `venv` to manage dependencies.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
Dependencies may vary per module. Navigate to the specific project folder and install the requirements:
```bash
cd 18-e2e-chatbot-mem0-tools-HITL-MCP-RAG
pip install -r requirements.txt
```

### 4. Environment Variables
Copy the `.env.example` file (if available in the module) to `.env` and add your API keys (OpenAI, Anthropic, HuggingFace, etc.):
```env
OPENAI_API_KEY="your_api_key_here"
ANTHROPIC_API_KEY="your_api_key_here"
TAVILY_API_KEY="your_api_key_here"
```

---

## 🤝 Contributing

This repository is continuously evolving! Contributions, bug reports, and feature requests are highly welcome. 

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License & Connect

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

<div align="center">
  
**Developed with 💡 by [Md Al Amin](https://github.com/MDalamin5)**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/mdalamin5/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MDalamin5)

If you find this repository helpful, don't forget to ⭐ star it!

</div>
