# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tianji (天机) is a Chinese social intelligence AI system focused on interpersonal communication, cultural etiquette, and emotional intelligence scenarios. The system provides guidance on social situations including toasting etiquette, gift-giving, conflict resolution, and general interpersonal communication within Chinese cultural contexts.

## Installation and Setup

### Environment Installation

```bash
pip install -e .
```

### Required Environment Configuration

Create a `.env` file in the project root with the following API keys:

```bash
# Essential API keys for basic functionality
ZHIPUAI_API_KEY=your_zhipu_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional configurations
HF_HOME='temp/huggingface_cache/'
HF_ENDPOINT='https://hf-mirror.com'
HF_TOKEN=your_huggingface_token
TAVILY_API_KEY=your_tavily_api_key_for_web_search
```

## Running the Application

### Main Demo Applications

```bash
# Multi-agent application with Streamlit UI
streamlit run run/demo_agent_metagpt.py

# RAG knowledge base application with LangChain
python run/demo_rag_langchain_onlinellm.py

# Simple prompt-based application
python run/demo_prompt.py

# Etiquette-specific RAG application
python run/demo_rag_langchain_etiquette.py
```

### Running Individual Tests

The project uses manual testing with standalone Python scripts:

```bash
# Test LLM models
python test/llm/chat/test_zhipuai.py
python test/llm/chat/test_openai.py

# Test RAG functionality
python test/knowledges/langchain/test_RAG_zhipuai.py
python test/knowledges/llamaindex/test_RAG_zhipuai_simple.py

# Test agent components
python test/agents/metagpt/answerBot_test_case.py
```

## Code Quality and Development

### Pre-commit Setup

The project uses pre-commit hooks for code formatting:

```bash
pip install pre-commit
pre-commit install

# Commits will automatically format code
git add .
git commit -m "your commit message"
```

## Architecture Overview

Tianji implements a multi-modal architecture with three main operational modes:

### 1. Agent System (`tianji/agents/metagpt_agents/`)

Multi-agent pipeline using MetaGPT framework:

- **IntentRecognition**: Classifies user queries into 7 social scenario categories
- **SceneRefinement**: Extracts missing contextual information from users  
- **AnswerBot**: Generates culturally appropriate responses
- **Searcher**: Optional web search enhancement using DuckDuckGo/Tavily APIs

Agent communication uses SharedDataSingleton for session-based state management with UUID-based user isolation.

### 2. Knowledge System (`tianji/knowledges/`)

RAG (Retrieval-Augmented Generation) implementation with both LangChain and LlamaIndex:

- **Vector Storage**: ChromaDB for document embeddings
- **LLM Integration**: ZhipuAI GLM-4-Flash and SiliconFlow Qwen2.5-7B-Instruct models
- **Knowledge Domains**: 7 specialized areas covering Chinese social etiquette scenarios

### 3. Prompt Engineering (`tianji/prompt/`)

Comprehensive prompt template library organized by scenario:

- **7 Main Categories**: Etiquette, Hospitality, Gifting, Wishes, Communication, Awkwardness, Conflict
- **Multi-Model Support**: Separate prompts for GPT, YiYan, and ZhiPu models
- **Interactive Games**: AI-powered social scenario simulations

### 4. Fine-tuning Infrastructure (`tianji/finetune/`)

Model customization using Transformers and XTuner frameworks:

- **LoRA/QLoRA Training**: Parameter-efficient fine-tuning
- **Supported Models**: Qwen2.5, Qwen2-7B, InternLM2
- **Social Intelligence Datasets**: Specialized training data for Chinese cultural contexts

## Data and Tools

### Core Datasets (`prebuilt-dataset/`)

- Social etiquette scenarios in JSON format
- RAG knowledge base in Chinese covering 7 social domains
- Fine-tuning datasets for specific social intelligence tasks

### Development Tools (`tools/`)

- **RAG Tools**: Data filtering, knowledge extraction, vector database construction
- **Fine-tuning Tools**: Data generation, cleaning, format conversion  
- **Prompt Tools**: Template validation, bulk processing, quality checking

## Key File Locations

### Configuration

- Environment variables: `.env` (create this file)
- Package configuration: `setup.py`
- Dependencies: `requirements.txt`

### Main Entry Points

- Agent demo: `run/demo_agent_metagpt.py`
- RAG demo: `run/demo_rag_langchain_onlinellm.py`
- Simple demo: `run/demo_prompt.py`

### Core Implementation

- Agent system: `tianji/agents/metagpt_agents/`
- Knowledge base: `tianji/knowledges/`
- Prompt templates: `tianji/prompt/`
- Fine-tuning: `tianji/finetune/`

### Testing

- Test files: `test/` directory with standalone Python scripts
- Manual execution required - no automated test runner

## Common Development Patterns

### LLM Model Integration

The system supports multiple LLM providers through unified interfaces:

- Custom LLM classes in `tianji/knowledges/langchain_onlinellm/models.py`
- Environment-based model selection
- Consistent API across different providers

### Data Processing Pipeline

Standard workflow for processing social intelligence data:

1. Raw data collection and filtering (`tools/rag/0-data_llm_filter.py`)
2. Knowledge extraction and clustering (`tools/rag/1-get_rag_knowledges.py`)  
3. Vector database construction and optimization
4. Quality assurance and testing

### Session Management

Multi-agent conversations maintain state through:

- SharedDataSingleton pattern for cross-agent communication
- UUID-based user session isolation
- Persistent context across agent handoffs

The system is designed to provide culturally sensitive social intelligence advice while supporting flexible deployment scenarios from lightweight prompt-based interactions to sophisticated multi-agent conversations with web-enhanced knowledge retrieval.
