#!/usr/bin/env python
# coding=utf-8
"""
ACE (Agentic Context Engineering) Examples

See README_ACE.md for detailed explanations of each section.

This script demonstrates:
1. Basic ACE usage
2. Playbook persistence
3. ACE as managed_agent
4. Planning integration
5. Individual role usage
6. Team learning with shared playbooks
7. Statistics & introspection
"""

import os
from smolagents import CodeAgent, ToolCallingAgent, InferenceClientModel, WebSearchTool, tool
from smolagents.ace import ACEAgent, ACEGenerator, ACEReflector, ACECurator, Playbook

# ============================================================
# SETUP
# ============================================================
# See README_ACE.md for model options
model = InferenceClientModel()

# ============================================================
# 1. BASIC USAGE
# ============================================================
print("=" * 70)
print("1. BASIC ACE USAGE")
print("=" * 70)

# ACE wraps the agent (CodeAgent, ToolCallingAgent, or any MultiStepAgent)
agent = CodeAgent(model=model, tools=[])
ace = ACEAgent(agent, auto_improve=True)

print(f"✓ ACEAgent created")
print(f"  - Agent type: {type(ace.agent).__name__}")
print(f"  - Auto-improve: {ace.auto_improve}")
print(f"  - Playbook entries: {len(ace.playbook)}")

# Run tasks - playbook evolves automatically
print("\n→ Running tasks...")
tasks = [
    "If I invest $1000 at 5% annual interest, how much will I have in 10 years?",
    "Calculate NPV: cash flows [100, 200, 300] at 8% discount rate",
]

for i, task in enumerate(tasks, 1):
    print(f"\n  Task {i}: {task[:50]}...")
    try:
        result = ace.run(task)
        print(f"  ✓ Result: {str(result)[:60]}...")
        print(f"  ✓ Playbook now: {len(ace.playbook)} entries")
    except Exception as e:
        print(f"  (Demo - playbook evolves with each task)")

# ============================================================
# 2. PLAYBOOK PERSISTENCE
# ============================================================
print("\n" + "=" * 70)
print("2. PLAYBOOK PERSISTENCE")
print("=" * 70)

# Save learned knowledge
playbook_file = "/tmp/finance_playbook.json"
ace.save_playbook(playbook_file)
print(f"✓ Saved playbook")
print(f"  - Version: {ace.playbook.version}")
print(f"  - Entries: {len(ace.playbook)}")

# Load into different agent type
new_agent = ToolCallingAgent(model=model, tools=[])
ace2 = ACEAgent.from_playbook(playbook_file, agent=new_agent)
print(f"\n✓ Loaded into {type(ace2.agent).__name__}")
print(f"  - Entries: {len(ace2.playbook)}")
print("  - Knowledge is portable across agent types")

# ============================================================
# 3. ACE AS MANAGED_AGENT
# ============================================================
print("\n" + "=" * 70)
print("3. ACE AS MANAGED_AGENT")
print("=" * 70)

# ACE can be a specialist in multi-agent systems
worker = CodeAgent(model=model, tools=[])
ace_worker = ACEAgent(
    worker,
    name="research_assistant",
    description="Learns from research tasks",
    auto_improve=True,
)

# Manager delegates to ACE worker
manager = CodeAgent(model=model, tools=[], managed_agents=[ace_worker])

print(f"✓ Multi-agent system")
print(f"  - Manager: {type(manager).__name__}")
print(f"  - Worker: {ace_worker.name}")
print("  - Manager delegates → Worker learns")

# ============================================================
# 4. PLANNING INTEGRATION
# ============================================================
print("\n" + "=" * 70)
print("4. PLANNING INTEGRATION")
print("=" * 70)

# ACE can reflect on planning steps
planning_agent = CodeAgent(model=model, tools=[], planning_interval=3)
ace_planner = ACEAgent(planning_agent, reflect_on_planning=True)

print(f"✓ ACE with planning")
print(f"  - Planning interval: {planning_agent.planning_interval} steps")
print(f"  - Reflects on plans: {ace_planner.reflect_on_planning}")

# ============================================================
# 5. INDIVIDUAL ROLES
# ============================================================
print("\n" + "=" * 70)
print("5. INDIVIDUAL ROLES")
print("=" * 70)

# Use roles separately for custom workflows
playbook = Playbook(name="custom", description="Step-by-step")

# Generator: Execute + capture trajectory
generator = ACEGenerator(
    agent=CodeAgent(model=model, tools=[]),
    playbook=playbook,
    use_step_callbacks=True,
)
print("✓ Generator: Execute + trajectory capture")

# Reflector: Analyze execution
reflector = ACEReflector(model=model)
print("✓ Reflector: Analyze + extract insights")

# Curator: Update playbook with dedup/pruning
curator = ACECurator(playbook=playbook, similarity_threshold=0.85, prune_threshold=-3)
print("✓ Curator: Deduplicate + prune + update playbook")

# ============================================================
# 6. STATISTICS & INTROSPECTION
# ============================================================
print("\n" + "=" * 70)
print("6. STATISTICS & INTROSPECTION")
print("=" * 70)

print("✓ Introspection methods:")
print(f"  - ace.stats()")
print(f"  - ace.show_playbook()")
print(f"  - ace.get_last_reflection()")
print(f"  - ace.get_last_curation_stats()")

print(f"\nExample stats:")
print(f"  - Runs: {ace.run_count}")
print(f"  - Improvements: {ace.improvement_count}")
print(f"  - Playbook entries: {len(ace.playbook)}")
print(f"  - Version: {ace.playbook.version}")

# ============================================================
# 7. SHARED PLAYBOOK - Team Learning
# ============================================================
print("\n" + "=" * 70)
print("7. SHARED PLAYBOOK")
print("=" * 70)

# Multiple agents share the same playbook = team learning
shared_playbook = Playbook(
    name="team_knowledge",
    description="Shared across the team"
)

agent1 = ACEAgent(
    CodeAgent(model=model, tools=[]),
    playbook=shared_playbook,
    name="agent1",
    auto_improve=True,
)

agent2 = ACEAgent(
    ToolCallingAgent(model=model, tools=[]),
    playbook=shared_playbook,
    name="agent2",
    auto_improve=True,
)

print(f"✓ Shared playbook: '{shared_playbook.name}'")
print(f"  - Agent1: {agent1.name}")
print(f"  - Agent2: {agent2.name}")
print(f"  - Same playbook: {id(agent1.playbook) == id(agent2.playbook)}")
print("  - Team learning enabled!")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("✨ ACE: Self-Improving Agents")
print("=" * 70)
print("""
ACE = Generator → Reflector → Curator

Benefits:
✓ Pure orchestration (no wrappers)
✓ Agent-agnostic (works with any agent type)
✓ Knowledge portability (save/load playbooks)
✓ Team learning (shared playbooks)
✓ Production-ready (semantic dedup, pruning)

See README_ACE.md for more details!
""")
print("=" * 70)
