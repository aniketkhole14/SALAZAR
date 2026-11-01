# SALAZAR

### Portable Offline AI Workstation

**Your AI. Your models. Your data. Your machine.**

SALAZAR is a portable, locally operated AI workstation that brings conversational AI, document intelligence, semantic search, and AI-assisted workflows into a single self-contained environment.

It is built around existing open-source models and inference runtimes. The focus is on the engineering required to **package, integrate, and operate** an AI ecosystem locally, rather than training a foundation model from scratch.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Bundled Models](#bundled-models)
- [How RAG Works](#how-rag-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Development](#development)
- [Design Philosophy](#design-philosophy)
- [Roadmap](#roadmap)
- [Current Limitations](#current-limitations)
- [Contributing](#contributing)
- [Project Status](#project-status)
- [License](#license)

---

## Overview

SALAZAR combines a desktop interface, a local backend, model inference, document retrieval, and workflow orchestration into one integrated application. It is designed to run from a portable USB drive, keeping application components, model assets, and runtime files together.

---

## Key Features

| Feature | Description |
|---|---|
| **Local AI inference** | Runs a quantized language model locally through a bundled llama.cpp-based runtime. |
| **Document intelligence** | Ingests documents, retrieves relevant passages, and generates answers grounded in their contents. |
| **Semantic retrieval** | Converts document passages into embeddings for similarity-based search. |
| **Workflow orchestration** | Specialized workflows for document research and code review. |
| **Conversation management** | Local conversation history stored by the backend. |
| **Portable desktop UI** | Electron-based desktop application. |
| **Bundled runtime** | Backend and inference components are packaged alongside the application. |

### Local language model inference

SALAZAR integrates a local language model through llama.cpp and communicates with the local inference backend without depending on a hosted LLM API for its core workflow.

**Current bundled model: SmolLM2 1.7B Instruct**

- **Format:** GGUF
- **Quantization:** Q4_K_M
- **Execution:** Local inference
- **Runtime:** Bundled llama.cpp-based executable

### Document intelligence and RAG

SALAZAR includes a Retrieval-Augmented Generation pipeline. Retrieved passages carry source references such as filenames and page numbers where available.

- **Embedding model:** BGE Small EN v1.5
- **Reranker:** MiniLM model assets are included. Runtime activation is not yet exposed as a verified capability.

### Workflow orchestration

Currently implemented workflows:

- **Document Research & Synthesis:** prepares a document-focused research request using the available retrieval pipeline.
- **Code Review & Explanation:** prepares a code-focused request for analysis and explanation.

The workflow interface prepares requests for submission through the chat interface.

### Conversation management

A local conversation storage component lets the backend maintain conversations, rather than relying entirely on temporary frontend state.

### Models registry

The **Models** page provides visibility into the model assets bundled with the workstation.

---

## Architecture

SALAZAR follows a modular architecture that separates the desktop interface, backend services, inference runtime, and retrieval pipeline.

```text
                  SALAZAR WORKSTATION
                 ┌────────────────────┐
                 │    Electron UI     │
                 │   React + Vite     │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │      FastAPI       │
                 │   Backend Layer    │
                 └─────────┬──────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌───────────┐ ┌───────────┐
        │   LLM    │ │    RAG    │ │ Workflows │
        │ Inference│ │ Pipeline  │ │   Agent   │
        └────┬─────┘ └─────┬─────┘ └───────────┘
             │             │
             ▼             ▼
        ┌──────────┐ ┌───────────┐
        │   GGUF   │ │ Embedding │
        │  Model   │ │   Model   │
        └──────────┘ └───────────┘
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Desktop application | Electron |
| Frontend | React, TypeScript |
| Frontend build | Vite |
| Backend | Python, FastAPI |
| LLM inference | llama.cpp |
| Model format | GGUF |
| Embeddings | BGE Small EN v1.5 |
| Reranker assets | MiniLM |
| Backend packaging | PyInstaller |
| Desktop packaging | electron-builder |

---

## Bundled Models

| Model | Purpose | Status |
|---|---|---|
| SmolLM2 1.7B Instruct | Language inference | Bundled |
| BGE Small EN v1.5 | Embeddings | Bundled |
| MiniLM | Reranker | Assets bundled |

---

## How RAG Works

1. Import a supported document.
2. Extract its textual content.
3. Divide the content into smaller chunks.
4. Generate embeddings.
5. Retrieve relevant passages for a query.
6. Supply the retrieved context to the language model.
7. Generate an answer grounded in the retrieved material.

---

## Project Structure

```text
SALAZAR/
├── app/
│   ├── api.py
│   ├── backend_entry.py
│   ├── bootstrap.py
│   ├── conversation_store.py
│   ├── rag_cli_handler.py
│   ├── runtime_discovery.py
│   ├── task_router.py
│   └── workflow_agent.py
├── frontend/
│   ├── electron/
│   │   └── main.cjs
│   ├── src/
│   │   ├── App.tsx
│   │   └── styles/
│   │       └── global.css
│   └── package.json
├── models/
│   ├── llm/
│   ├── embeddings/
│   └── reranker/
├── runtime/
│   ├── backend/
│   ├── rag/
│   └── llama.cpp/
├── scripts/
│   ├── ask_salazar.py
│   ├── ingest_salazar.py
│   └── test_workflow_agent.py
├── Launch-SALAZAR.sh
├── SALAZAR.desktop
├── requirements-runtime.txt
├── requirements-rag.txt
├── salazar-backend.spec
├── salazar-rag.spec
└── .gitignore
```

> [!NOTE]
> Some `runtime/` and `models/` directories are intentionally excluded from the Git repository because they contain bundled binaries, third-party runtime files, or large model assets.

---

## Getting Started

### Supported platform

**Linux x86-64.** The current portable desktop build has been tested on Fedora Linux.

### Run from the USB drive

1. Connect the USB drive.
2. Open the mounted `SALAZAR` directory.
3. Launch `SALAZAR.desktop` from your desktop environment, **or** run the shell launcher from a terminal:

```bash
cd /path/to/SALAZAR
bash ./Launch-SALAZAR.sh
```

The launcher resolves the application directory and sets the SALAZAR root environment variable before starting the packaged desktop application.

### Portability considerations

SALAZAR bundles its application, backend executable, inference components, and model assets. However:

- The host Linux distribution must provide compatible system libraries.
- The USB filesystem must permit execution.

> [!WARNING]
> The current build is **not** a universal, dependency-free Linux binary.

---

## Development

### Requirements

- Python 3
- Node.js and npm
- Linux development environment
- Git

### Dependency manifests

| Scope | File |
|---|---|
| Runtime | `requirements-runtime.txt` |
| RAG | `requirements-rag.txt` |
| Frontend | `frontend/package.json` |

PyInstaller specifications for the backend and RAG executables are provided in `salazar-backend.spec` and `salazar-rag.spec`.

> [!NOTE]
> Development and packaging commands may vary by platform. The currently verified portable release is the packaged Linux build.

---

## Design Philosophy

1. **Local-first architecture:** core inference and document retrieval are designed to run locally.
2. **Model independence:** SALAZAR is an application ecosystem around existing models, not a new foundation model.
3. **Portability:** application components, model assets, and runtime dependencies travel together.
4. **Modularity:** inference, retrieval, workflow orchestration, and the UI are separate components that can evolve independently.
5. **User control:** the long-term goal is to let users manage their own model files, configurations, and local AI environment.

---

## Roadmap

### Completed

- [x] Electron desktop application
- [x] React and TypeScript frontend
- [x] FastAPI backend integration
- [x] Local llama.cpp inference integration
- [x] GGUF model integration
- [x] Document ingestion and retrieval pipeline
- [x] Embedding model integration
- [x] Conversation storage
- [x] Workflow orchestration
- [x] Models registry interface
- [x] PyInstaller backend packaging
- [x] Linux portable launcher
- [x] USB-based launch and reconnect testing

### Planned

- [ ] Windows compatibility and packaging
- [ ] User-importable GGUF models
- [ ] Custom model directory management
- [ ] Model metadata detection and validation
- [ ] Model selection and switching
- [ ] Runtime configuration for different hardware
- [ ] Improved model lifecycle management
- [ ] Expanded workflow system
- [ ] More advanced RAG configuration
- [ ] Runtime status monitoring

### Planned: user-managed model ecosystem

A major planned feature is importing compatible GGUF models from external repositories:

1. Download a compatible GGUF model.
2. Import or select the model file inside SALAZAR.
3. Validate its metadata and compatibility.
4. Register it in the local model library.
5. Select it for inference.

> [!IMPORTANT]
> This feature is planned and is **not yet available** in the current build.

---

## Current Limitations

- Linux x86-64 is the only verified desktop platform.
- Windows support is not yet implemented.
- Custom user-imported GGUF model management is not yet implemented.
- The Models page displays bundled assets and does not provide model switching.
- MiniLM reranker assets are bundled, but active runtime integration has not been verified.
- Compatibility with other Linux distributions depends on system libraries and mount configuration.
- Performance depends on the host CPU, available RAM, and model configuration.

---

## Contributing

SALAZAR is an evolving engineering project. Contributions, bug reports, architectural suggestions, and improvements are welcome.

When contributing:

1. Keep modules focused and maintainable.
2. Avoid committing model weights or generated binaries.
3. Keep personal documents and local runtime data out of version control.
4. Include tests when modifying backend behavior.
5. Document platform-specific changes.

---

## Project Status

| | |
|---|---|
| **Current milestone** | Portable Linux Workstation |
| **Development branch** | `development` |
| **Platform** | Linux x86-64 |

SALAZAR is under active development. Features listed in the roadmap are planned and should not be interpreted as currently available functionality.

---

## License

License information has not yet been specified. Please check this repository for future licensing updates before redistributing or modifying the project.

---

**SALAZAR: Intelligence without being tied to a machine.**
