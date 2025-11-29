#!/usr/bin/env python
# coding=utf-8
"""
ACE (Agentic Context Engineering) Example

Demonstrates self-improving agents through evolving playbooks.
Uses smolagents' native mechanisms - no wrappers, no monkey-patching.

Features shown:
1. Basic ACE usage with any smolagents agent
2. Playbook persistence and loading
3. ACE as managed_agent in multi-agent systems
4. Planning integration with planning_interval
5. Individual role usage (Generator, Reflector, Curator)
"""

from smolagents import CodeAgent, ToolCallingAgent, HfApiModel, WebSearchTool
from smolagents.ace import ACEAgent, ACEGenerator, ACEReflector, ACECurator, Playbook

# ============================================================
# 1. BASIC USAGE - ACE orchestrates any smolagents agent
# ============================================================
print("=" * 60)
print("1. BASIC USAGE")
print("=" * 60)

model = HfApiModel("meta-llama/Llama-3.3-70B-Instruct")

# Create any smolagents agent - ACE works with all types
agent = CodeAgent(model=model, tools=[])
# Or: agent = ToolCallingAgent(model=model, tools=[])

# ACE orchestrates the agent using native smolagents mechanisms:
# - `instructions` for playbook injection
# - `step_callbacks` for real-time trajectory capture
ace = ACEAgent(agent, auto_improve=True)

print(f"Created: {ace}")
print(f"Underlying agent: {type(ace.agent).__name__}")

# Run tasks - playbook evolves automatically
tasks = [
    "Calculate compound interest for $1000 at 5% over 10 years",
    "What's the NPV of cash flows [100, 200, 300] at 8% discount?",
]

for task in tasks:
    print(f"\nTask: {task[:50]}...")
    result = ace.run(task)
    print(f"Playbook: {len(ace.playbook)} entries")

# ============================================================
# 2. PLAYBOOK PERSISTENCE
# ============================================================
print("\n" + "=" * 60)
print("2. PLAYBOOK PERSISTENCE")
print("=" * 60)

# Save learned strategies
ace.save_playbook("finance_playbook.json")
print("Saved playbook to finance_playbook.json")

# Load in new session with different agent type
new_agent = ToolCallingAgent(model=model, tools=[])
ace2 = ACEAgent.from_playbook("finance_playbook.json", agent=new_agent)
print(f"Loaded playbook v{ace2.playbook.version} into {type(ace2.agent).__name__}")

# ============================================================
# 3. ACE AS MANAGED_AGENT (Multi-Agent Systems)
# ============================================================
print("\n" + "=" * 60)
print("3. ACE AS MANAGED_AGENT")
print("=" * 60)

# ACE can be a managed_agent - it has name, description, and is callable
worker = CodeAgent(model=model, tools=[WebSearchTool()])
ace_worker = ACEAgent(
    worker,
    name="learning_researcher",
    description="A researcher that learns and improves from each task",
    auto_improve=True,
)

# Manager can delegate to ACE agent
manager = CodeAgent(
    model=model,
    tools=[],
    managed_agents=[ace_worker],  # ACE as managed agent!
)

print(f"Manager has managed_agent: {ace_worker.name}")
print(f"  Description: {ace_worker.description}")

# Manager delegates, ACE learns
# result = manager.run("Research current AI trends and summarize")

# ============================================================
# 4. PLANNING INTEGRATION
# ============================================================
print("\n" + "=" * 60)
print("4. PLANNING INTEGRATION")
print("=" * 60)

# ACE integrates with smolagents' planning system
planning_agent = CodeAgent(
    model=model,
    tools=[],
    planning_interval=3,  # Plan every 3 steps
)

ace_planner = ACEAgent(
    planning_agent,
    reflect_on_planning=True,  # Reflect after PlanningSteps
)

print(f"Planning agent with ACE: planning_interval={planning_agent.planning_interval}")
print("ACE will reflect after each PlanningStep")

# ============================================================
# 5. INDIVIDUAL ROLES (Advanced Usage)
# ============================================================
print("\n" + "=" * 60)
print("5. INDIVIDUAL ROLES")
print("=" * 60)

# Use ACE roles separately for custom workflows
playbook = Playbook()

# Generator: Execute with playbook injection
generator = ACEGenerator(
    agent=CodeAgent(model=model, tools=[]),
    playbook=playbook,
    use_step_callbacks=True,  # Real-time trajectory capture
)

gen_result = generator.run("Calculate 2^10")
print(f"Generator result: {gen_result['result']}")
print(f"Trajectory steps: {len(gen_result['trajectory'])}")

# Reflector: Analyze trajectory
reflector = ACEReflector(model=model)
reflection = reflector.reflect(gen_result)
print(f"Reflection success: {reflection.get('overall_success')}")
print(f"New insights: {len(reflection.get('new_insights', []))}")

# Curator: Update playbook with semantic dedup
curator = ACECurator(playbook=playbook, similarity_threshold=0.85)
_, stats = curator.curate(reflection)
print(f"Curation: +{stats['insights_added']} insights")

# ============================================================
# 6. STATISTICS & INTROSPECTION
# ============================================================
print("\n" + "=" * 60)
print("6. STATISTICS")
print("=" * 60)

print(f"ACE Stats: {ace.stats()}")
print(f"\nPlaybook Content:")
print(ace.show_playbook())

# ============================================================
# 7. SHARED PLAYBOOK ACROSS AGENTS
# ============================================================
print("\n" + "=" * 60)
print("7. SHARED PLAYBOOK")
print("=" * 60)

# Multiple agents can share and contribute to the same playbook
shared_playbook = Playbook(name="team_knowledge", description="Shared team strategies")

agent1 = ACEAgent(CodeAgent(model=model, tools=[]), playbook=shared_playbook)
agent2 = ACEAgent(ToolCallingAgent(model=model, tools=[]), playbook=shared_playbook)

print(f"Agent1 and Agent2 share: {shared_playbook.name}")
print("Both agents contribute to and benefit from the same playbook!")

# You can also merge playbooks
other_playbook = Playbook.load("finance_playbook.json") if False else Playbook()
merged = ace.merge_playbook(other_playbook)
print(f"Merged {merged} entries from another playbook")

print("\n" + "=" * 60)
print("ACE Module - MCP Birthday Gift Complete!")
print("=" * 60)
