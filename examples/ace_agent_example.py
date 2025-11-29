#!/usr/bin/env python
# coding=utf-8
"""
ACE (Agentic Context Engineering) Example

Demonstrates self-improving agents through evolving playbooks.
Uses smolagents' native mechanisms - no wrappers, no monkey-patching.

This example shows:
1. Basic ACE usage with any smolagents agent
2. How the Playbook evolves automatically with auto_improve
3. Playbook persistence and loading
4. ACE as managed_agent in multi-agent systems
5. Planning integration with planning_interval
6. Individual role usage (Generator, Reflector, Curator)
7. Shared playbooks across multiple agents
8. Introspection and statistics

Key architecture:
- ACEAgent wraps any smolagents agent (CodeAgent, ToolCallingAgent, etc.)
- Generator: Injects playbook via `instructions` + captures trajectory
- Reflector: Analyzes execution to extract learnings
- Curator: Updates playbook with semantic deduplication & pruning
- Playbook: Evolving knowledge base (STRATEGIES, FORMULAS, MISTAKES)
"""

import os
from smolagents import CodeAgent, ToolCallingAgent, InferenceClientModel, WebSearchTool, tool
from smolagents.ace import ACEAgent, ACEGenerator, ACEReflector, ACECurator, Playbook

# ============================================================
# SETUP - Choose your model
# ============================================================
# Option 1: HuggingFace Inference (free, requires HF token)
model = InferenceClientModel()

# Option 2: Local model via transformers
# from smolagents import TransformersModel
# model = TransformersModel(model_id="Qwen/Qwen2.5-72B-Instruct", max_new_tokens=2048)

# Option 3: Any LLM via LiteLLM
# from smolagents import LiteLLMModel
# os.environ["ANTHROPIC_API_KEY"] = "your-key"
# model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20241022", temperature=0.2)

# ============================================================
# 1. BASIC USAGE - ACE orchestrates any smolagents agent
# ============================================================
print("=" * 70)
print("1. BASIC ACE USAGE - Pure Orchestration")
print("=" * 70)
print("""
ACE doesn't wrap agents - it orchestrates them using native mechanisms:
- `instructions`: Playbook injected as context to the agent
- `step_callbacks`: Real-time trajectory capture for reflection
- Fully compatible with any smolagents agent type
""")

# Create a simple agent - ACE works with both CodeAgent and ToolCallingAgent
agent = CodeAgent(model=model, tools=[])

# ACE wraps it - orchestrating the three roles (Generator -> Reflector -> Curator)
ace = ACEAgent(
    agent,
    auto_improve=True,  # Automatically reflect and update playbook after each run
    reflect_on_planning=False,  # Set to True if agent has planning_interval
)

print(f"✓ Created ACEAgent")
print(f"  - Underlying agent: {type(ace.agent).__name__}")
print(f"  - Auto-improve enabled: {ace.auto_improve}")
print(f"  - Initial playbook entries: {len(ace.playbook)}")

# Run simple financial tasks - playbook evolves automatically
print("\n📚 Running tasks to build playbook knowledge...")
tasks = [
    "If I invest $1000 at 5% annual interest, how much will I have in 10 years?",
    "Calculate NPV: cash flows are [100, 200, 300], discount rate is 8%",
    "What's the monthly payment on a $300k mortgage at 4% APR over 30 years?",
]

for i, task in enumerate(tasks, 1):
    print(f"\n  Task {i}: {task[:60]}...")
    try:
        result = ace.run(task)
        print(f"    ✓ Result: {str(result)[:80]}...")
        print(f"    ✓ Playbook now has {len(ace.playbook)} entries")
    except Exception as e:
        print(f"    (Task demo - showing how playbook evolves)")

# ============================================================
# 2. PLAYBOOK PERSISTENCE & LOADING
# ============================================================
print("\n" + "=" * 70)
print("2. PLAYBOOK PERSISTENCE - Save & Load Knowledge")
print("=" * 70)

# Save the evolved playbook
playbook_file = "/tmp/finance_playbook.json"
ace.save_playbook(playbook_file)
print(f"✓ Saved playbook to {playbook_file}")
print(f"  - Playbook version: {ace.playbook.version}")
print(f"  - Total entries: {len(ace.playbook)}")

# Load the playbook into a new agent (different type!)
# This shows playbooks are agent-agnostic
new_agent = ToolCallingAgent(model=model, tools=[])
ace2 = ACEAgent.from_playbook(playbook_file, agent=new_agent)
print(f"\n✓ Loaded playbook into {type(ace2.agent).__name__} (different agent type!)")
print(f"  - Playbook version: {ace2.playbook.version}")
print(f"  - Entries transferred: {len(ace2.playbook)}")
print("  - This shows knowledge is portable across agent types")

# ============================================================
# 3. ACE AS MANAGED_AGENT (Multi-Agent Systems)
# ============================================================
print("\n" + "=" * 70)
print("3. ACE AS MANAGED_AGENT - Self-Improving in Teams")
print("=" * 70)
print("""
ACE agents can be managed_agents in multi-agent systems.
Manager delegates subtasks to ACE, which learns from each task!
""")

# Create a worker agent that learns
worker = CodeAgent(model=model, tools=[])  # No WebSearchTool for demo

# Wrap it with ACE - give it a name and description for multi-agent use
ace_worker = ACEAgent(
    worker,
    name="research_assistant",
    description="A specialist that learns and improves from research tasks",
    auto_improve=True,
)

# Create a manager that delegates to the ACE worker
manager = CodeAgent(
    model=model,
    tools=[],
    managed_agents=[ace_worker],  # ACE worker as managed agent
)

print(f"✓ Created multi-agent system")
print(f"  - Manager: {type(manager).__name__}")
print(f"  - Managed agent: {ace_worker.name}")
print(f"  - Worker description: {ace_worker.description}")
print(f"  - Worker will auto-improve from delegated tasks")

print("\nNote: Manager could delegate like:")
print('  manager.run("Research X and summarize")')
print("  → Delegates to ace_worker")
print("  → ACE learns from the research task")
print("  → Future research tasks benefit from learnings!")

# ============================================================
# 4. PLANNING INTEGRATION
# ============================================================
print("\n" + "=" * 70)
print("4. PLANNING INTEGRATION - Reflecting on Execution Plans")
print("=" * 70)
print("""
ACE can integrate with smolagents' planning system.
When reflect_on_planning=True, ACE reflects on PlanningSteps.
This helps agents improve their planning strategies!
""")

# Create an agent with planning enabled
planning_agent = CodeAgent(
    model=model,
    tools=[],
    planning_interval=3,  # Create a plan every 3 steps
)

# Wrap with ACE that reflects on planning steps
ace_planner = ACEAgent(
    planning_agent,
    reflect_on_planning=True,  # Extract learnings from planning steps
)

print(f"✓ Created planning agent with ACE")
print(f"  - Planning interval: {planning_agent.planning_interval} steps")
print(f"  - Reflect on planning: {ace_planner.reflect_on_planning}")
print(f"  - Playbook will capture planning insights")

# ============================================================
# 5. INDIVIDUAL ROLES - Advanced Custom Workflows
# ============================================================
print("\n" + "=" * 70)
print("5. INDIVIDUAL ROLES - Fine-Grained Control")
print("=" * 70)
print("""
For advanced use cases, use the three roles individually:
- ACEGenerator: Execute task + capture trajectory
- ACEReflector: Analyze trajectory + extract insights
- ACECurator: Deduplicate + prune + update playbook
""")

# Create an empty playbook
playbook = Playbook(name="custom_workflow", description="Built step-by-step")

# Role 1: Generator - Execute with playbook context
print("\n📌 GENERATOR: Execute task + capture trajectory")
generator = ACEGenerator(
    agent=CodeAgent(model=model, tools=[]),
    playbook=playbook,
    use_step_callbacks=True,  # Capture steps in real-time
)

print("  (Would execute a task and capture trajectory)")
print("  gen_result = generator.run('Calculate 2^10')")
print("  → Returns: {'result': ..., 'trajectory': [...]}")

# Role 2: Reflector - Analyze execution
print("\n📌 REFLECTOR: Analyze trajectory + extract insights")
reflector = ACEReflector(model=model)

print("  (Would analyze the trajectory)")
print("  reflection = reflector.reflect(gen_result)")
print("  → Returns: {'overall_success': bool, 'strategy_feedback': [...], 'new_insights': [...]}")

# Role 3: Curator - Update playbook intelligently
print("\n📌 CURATOR: Add insights + deduplicate + prune")
curator = ACECurator(
    playbook=playbook,
    similarity_threshold=0.85,  # Semantic similarity for dedup
    prune_threshold=-3,  # Remove entries with score < -3
)

print("  (Would update playbook with semantic deduplication)")
print("  updated_playbook, stats = curator.curate(reflection)")
print("  → Updates playbook with intelligent deduplication")
print("  → Prunes harmful entries")
print("  → Returns statistics on what was added/pruned")

print(f"\nCustom workflow allows fine-grained control over each ACE stage!")

# ============================================================
# 6. STATISTICS & INTROSPECTION
# ============================================================
print("\n" + "=" * 70)
print("6. STATISTICS & INTROSPECTION")
print("=" * 70)

print("✓ Available introspection methods:")
print(f"  - ace.stats(): {{'runs': X, 'improvements': Y, 'playbook_size': Z}}")
print(f"  - ace.show_playbook(): Display playbook content")
print(f"  - ace.get_last_reflection(): Last reflection insights")
print(f"  - ace.get_last_curation_stats(): Last curation changes")
print(f"  - ace.get_last_trajectory(): Last execution trajectory")

print(f"\nExample stats from our first ACE agent:")
print(f"  - Runs: {ace.run_count}")
print(f"  - Improvements: {ace.improvement_count}")
print(f"  - Playbook entries: {len(ace.playbook)}")
print(f"  - Playbook version: {ace.playbook.version}")

# ============================================================
# 7. SHARED PLAYBOOK ACROSS AGENTS
# ============================================================
print("\n" + "=" * 70)
print("7. SHARED PLAYBOOK - Team Learning")
print("=" * 70)
print("""
Multiple agents can share and contribute to the same playbook!
This enables team learning - collective intelligence across agents.
""")

# Create a shared playbook
shared_playbook = Playbook(
    name="team_finance_knowledge",
    description="Shared financial calculation strategies"
)

# Multiple agents contribute to the same playbook
agent1 = ACEAgent(
    CodeAgent(model=model, tools=[]),
    playbook=shared_playbook,
    name="calculator_agent",
    auto_improve=True,
)

agent2 = ACEAgent(
    ToolCallingAgent(model=model, tools=[]),
    playbook=shared_playbook,
    name="calculator_agent_v2",
    auto_improve=True,
)

print(f"✓ Created shared playbook: '{shared_playbook.name}'")
print(f"  - Agent1: {agent1.name} ({type(agent1.agent).__name__})")
print(f"  - Agent2: {agent2.name} ({type(agent2.agent).__name__})")
print(f"  - Both agents share learning: {id(agent1.playbook) == id(agent2.playbook)}")

print("\nWhen Agent1 learns something new:")
print("  → Entry added to shared_playbook")
print("  → Agent2 benefits immediately!")
print("  → Collective intelligence grows")

# Merge playbooks from different teams
if os.path.exists(playbook_file):
    other_playbook = Playbook.load(playbook_file)
    merged_count = ace.merge_playbook(other_playbook)
    print(f"\n✓ Merged {merged_count} entries from another playbook")
    print("  - Useful for combining team knowledge")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("✨ ACE: Self-Improving Agents through Orchestration")
print("=" * 70)
print("""
ACE enhances any smolagents agent with self-improvement:

1. GENERATOR  → Execute task with playbook as context
2. REFLECTOR → Analyze execution and extract insights
3. CURATOR   → Update playbook with intelligent deduplication

Key benefits:
✓ Pure orchestration - no wrappers, uses native mechanisms
✓ Agent-agnostic - works with CodeAgent, ToolCallingAgent, etc.
✓ Knowledge portability - save/load playbooks across sessions
✓ Team learning - share playbooks across multiple agents
✓ Production-ready - semantic similarity, pruning, persistence

Use cases:
→ Iterative tasks where agents improve over time
→ Multi-agent teams that learn collectively
→ Long-running agents that accumulate expertise
→ Research/evaluation of agent learning
""")
print("=" * 70)
