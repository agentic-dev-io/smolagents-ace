# ACE (Agentic Context Engineering) Examples

This directory contains examples of ACE in action - demonstrating how agents improve through evolving playbooks using pure orchestration with native smolagents mechanisms.

## Overview

ACE enhances any smolagents agent with self-improvement capabilities:

1. **Generator** - Executes tasks with playbook context injected via `instructions`
2. **Reflector** - Analyzes execution trajectories and extracts learnings
3. **Curator** - Updates playbook with intelligent deduplication and pruning

The playbook is an evolving knowledge base organized into:
- **STRATEGIES & INSIGHTS** - Proven approaches for solving tasks
- **FORMULAS & CALCULATIONS** - Useful equations and calculations
- **COMMON MISTAKES** - Pitfalls to avoid

## Key Files

### `ace_agent_example.py`
The main demonstration of ACE capabilities:

1. **Basic Usage** - Simple ACE setup with auto-improve
   - Create an ACEAgent wrapping any smolagents agent
   - Run tasks and watch playbook evolve automatically

2. **Playbook Persistence** - Save and load knowledge
   - Save evolved playbook to JSON
   - Load into new agents (works with different agent types!)
   - Knowledge is portable across sessions

3. **Managed Agents** - ACE in multi-agent systems
   - Use ACE as a `managed_agent` in team hierarchies
   - Manager delegates tasks, ACE learns from them
   - Enables specialized learning agents

4. **Planning Integration** - Learning from planning steps
   - Set `reflect_on_planning=True` for agents with `planning_interval`
   - Extract insights from planning trajectories
   - Improve planning strategies over time

5. **Individual Roles** - Fine-grained control
   - Use Generator, Reflector, Curator separately
   - Build custom ACE workflows
   - Integrate with existing pipelines

6. **Team Learning** - Shared playbooks across agents
   - Multiple agents contribute to same playbook
   - Collective intelligence grows
   - Merge playbooks from different teams

7. **Statistics & Introspection**
   - `ace.stats()` - runs, improvements, playbook size
   - `ace.show_playbook()` - view learned knowledge
   - `ace.get_last_reflection()` - insights from last run
   - `ace.get_last_curation_stats()` - what changed

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ACEAgent (Orchestrator)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │ ACEGenerator│ ──► │ACEReflector │ ──► │ ACECurator  │       │
│   │             │     │             │     │             │       │
│   │ instructions│     │ Analyze     │     │ Delta       │       │
│   │ + callbacks │     │ trajectory  │     │ updates     │       │
│   └──────┬──────┘     └─────────────┘     └─────────────┘       │
│          │                                       │               │
│          ▼                                       ▼               │
│   ┌─────────────┐              ┌─────────────────────┐          │
│   │ CodeAgent / │              │      PLAYBOOK       │          │
│   │ToolCalling  │◄─────────────│ (via instructions)  │          │
│   └─────────────┘              └─────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Use Cases

### ✅ Good fits for ACE:
- **Iterative tasks** - Run similar tasks repeatedly, improve each time
- **Multi-agent teams** - Agents learn collectively from shared playbooks
- **Long-running systems** - Agents that accumulate expertise over time
- **Research/evaluation** - Measure agent learning progress
- **Specialized workflows** - Teams with specific roles that improve together

### ❌ Not ideal for ACE:
- **One-shot tasks** - No opportunity for learning
- **Cost-sensitive** - Reflection adds LLM calls
- **Determinism required** - Learning introduces variability

## Quick Start

```python
from smolagents import CodeAgent, InferenceClientModel
from smolagents.ace import ACEAgent

# 1. Create a base agent
model = InferenceClientModel()
agent = CodeAgent(model=model, tools=[])

# 2. Wrap with ACE
ace = ACEAgent(agent, auto_improve=True)

# 3. Run tasks - playbook evolves automatically
result = ace.run("Your task here")

# 4. Check what was learned
print(ace.show_playbook())
print(ace.stats())
```

## Model Options

Choose which LLM to use:

### HuggingFace Inference (Free)
```python
from smolagents import InferenceClientModel
model = InferenceClientModel()
```

### Local Model via Transformers
```python
from smolagents import TransformersModel
model = TransformersModel(
    model_id="Qwen/Qwen2.5-72B-Instruct",
    max_new_tokens=2048
)
```

### Any LLM via LiteLLM
```python
from smolagents import LiteLLMModel
model = LiteLLMModel(
    model_id="anthropic/claude-3-5-sonnet-20241022",
    temperature=0.2
)
```

## Key Concepts

### Playbook Entries
Each entry in the playbook has:
- **ID** - Unique identifier (str-00001, cal-00002, mis-00003)
- **Section** - STRATEGIES, FORMULAS, or MISTAKES
- **Content** - The actual knowledge
- **Helpful Count** - How many times it helped
- **Harmful Count** - How many times it hurt

Score = helpful_count - harmful_count

### Native Mechanisms
ACE uses pure smolagents features:
- **`instructions`** - Playbook injected as context (no prompt hacking)
- **`step_callbacks`** - Real-time trajectory capture
- **`managed_agents`** - ACE can be a specialist in teams
- **`planning_interval`** - ACE can reflect on planning steps

### Reflection
After each task, ACE:
1. Analyzes what worked and what didn't
2. Extracts insights about strategies used
3. Updates playbook with new learnings
4. Prunes harmful entries

### Semantic Deduplication
Curator uses embedding models to:
- Avoid duplicate knowledge
- Merge similar insights
- Keep playbook clean and efficient

## Running Examples

```bash
# Run the main example (shows features, not full execution)
python examples/ace_agent_example.py

# All other examples in the directory work as usual
python examples/multiple_tools.py
python examples/sandboxed_execution.py
# etc.
```

## Integration with smolagents

ACE is a pure orchestration layer - it doesn't replace smolagents mechanisms, it enhances them:

- Works with **any agent type** (CodeAgent, ToolCallingAgent, custom MultiStepAgent)
- Works with **any model** (HF, OpenAI, LiteLLM, Bedrock, etc.)
- Works with **any tools** (built-in tools, custom tools, MCP servers, LangChain tools)
- Fully compatible with **managed_agents**, **planning_interval**, **step_callbacks**

## Advanced Topics

### Custom Reflection Prompts
Customize what Reflector looks for:
```python
reflector = ACEReflector(
    model=model,
    reflection_prompt="Your custom analysis prompt..."
)
```

### Semantic Similarity Tuning
Adjust deduplication sensitivity:
```python
curator = ACECurator(
    playbook=playbook,
    similarity_threshold=0.85,  # 0-1, higher = stricter dedup
    prune_threshold=-3,  # Remove entries with score < -3
)
```

### Shared Playbooks
Enable team learning:
```python
shared_playbook = Playbook(
    name="team_knowledge",
    description="Shared across the team"
)

agent1 = ACEAgent(worker1, playbook=shared_playbook)
agent2 = ACEAgent(worker2, playbook=shared_playbook)

# Both agents learn together!
```

### Playbook Merging
Combine knowledge from different sources:
```python
ace1.merge_playbook(other_playbook)
```

## Learning More

- See `src/smolagents/ace/` for implementation details
- Main components: `ace_agent.py`, `generator.py`, `reflector.py`, `curator.py`, `playbook.py`
- Integration with smolagents: uses native `instructions`, `step_callbacks`, `managed_agents`

## Troubleshooting

**Playbook not evolving?**
- Check `auto_improve=True`
- Verify Reflector has a model configured
- Check logs for reflection errors

**Memory/Cost issues?**
- Reduce `planning_interval` for less frequent reflection
- Set `auto_improve=False` and call `improve()` manually
- Use smaller embedding models in Curator

**Playbook quality degrading?**
- Lower `similarity_threshold` for stricter deduplication
- Increase `prune_threshold` to remove unhelpful entries faster
- Monitor with `ace.show_playbook()` and `ace.get_last_curation_stats()`

---

**Remember:** ACE enhances smolagents agents, it doesn't replace them. Use it when agent learning adds value to your use case!
