# AI Research Implementation Team

## Introduction
The AI Research Implementation Team is dedicated to advancing artificial intelligence technologies and translating cutting-edge research into practical applications. Our team of scientists and engineers works collaboratively to design, develop, and deploy innovative AI solutions that can solve complex problems across various domains.

## Team Objectives
- **Mission:** To leverage AI research for tangible benefits, enhancing productivity and efficiency across industries.
- **Goals:**
  - Develop state-of-the-art AI models.
  - Implement real-world applications of AI research findings.
  - Foster collaboration within the AI research community.

## Getting Involved
We welcome contributions from researchers, developers, and enthusiasts.

## Setup

```
pip install -r requirements.txt
cp env.example .env
# edit .env: set LLM_PROVIDER (groq | openrouter | openai) and the matching API key
python main.py path/to/paper.pdf        # CLI
python main.py serve                    # REST API on :8000 (see env.example for auth/upload config)
```

See `env.example` for the full list of environment variables, including
per-agent model overrides and REST API configuration (`API_AUTH_TOKEN`,
`UPLOADS_DIR`). See `FIXES.md` for a detailed record of recent bug fixes to
this codebase.

## Contact Information
For inquiries or collaboration opportunities, please reach out to the team lead at [obedienceadara@gmail.com].
