# Agentic-Profile-Matching

## Run

Install dependencies and add your OpenRouter key to `.env`:

```bash
python3 -m pip install -r requirements.txt
OPEN_ROUTER_MODEL=openrouter/free
```

Then start the agent and enter the job description when prompted:

```bash
python3 matching_agent.py
# JD: python developer
```


```bash
python3 matching_agent.py
# JD: Product manager with a background in fintech
```

```bash
python3 matching_agent.py
# JD: Kafka with 5+ years of experience
```


```bash
python3 matching_agent.py
# JD: fintech domain exeperts
```