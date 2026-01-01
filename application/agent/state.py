"""
MODULE: Agent State
DESCRIPTION: 
    Defines the schema of the Agent's Working Memory.
    Uses 'TypedDict' for strict typing and 'Annotated' for reducer logic.

ARCHITECTURE:
    - Type: Append-Only Log (Steps)
    - Persistence: Passed between every node in the graph.
"""

import operator
from typing import Annotated, List, TypedDict, Union

class AgentState(TypedDict):
    """
    The Brain's Working Memory.
    
    FIELDS:
    - question: The user's query (mutable - can be rewritten by the agent).
    - cypher_query: The generated database query.
    - context: Documents/Facts retrieved from the graph/vector store.
    - answer: The final generated response.
    - grade: 'useful' or 'not_useful' (signals the conditional edge).
    - steps: A log of actions taken. 'operator.add' ensures we APPEND, not OVERWRITE.
    - attempts: Counter to prevent infinite loops.
    """
    question: str
    cypher_query: str
    context: List[dict]
    answer: str
    grade: str
    
    # --- THE SAFETY MECHANISM ---
    # Annotated[List[str], operator.add] means:
    # When a node returns {"steps": ["new_step"]}, it adds to the list:
    # ["old_step"] + ["new_step"] = ["old_step", "new_step"]
    # This prevents the agent from "forgetting" what it just did.
    steps: Annotated[List[str], operator.add]
    
    attempts: int