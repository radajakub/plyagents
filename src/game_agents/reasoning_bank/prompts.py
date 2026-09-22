# Ported from the memory_is_dynamic branch of the WebArena agent's system-message assembly
# (WebArena/agents/legacy/agent.py), where the retrieved items are appended to the system prompt.
# The wording is upstream's, re-domained from web navigation to turn-based games.

RETRIEVED_EXPERIENCE = """# Retrieved experience for the current state
The memory items below were retrieved specifically for the current observation and apply only to this turn. For each item, check that its conditions match the current state, player role, and legal moves. Ignore an item when they do not match. Use its strategic principle rather than copying literal move labels, verify that every proposed move is currently legal, and prioritize immediate wins, forced blocks, and prevention of forced losses over general advice."""
