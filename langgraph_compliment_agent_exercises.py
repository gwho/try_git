"""
LangGraph Compliment Agent - Practice Exercises
================================================
Mini-projects to test and expand your understanding!

Instructions:
1. Complete each exercise by modifying the original langgraph_compliment_agent.py
2. Test your changes by running the file
3. Try to solve without looking at the tutorial comments first
4. Each exercise builds on the previous one

Prerequisites: Complete langgraph_compliment_agent.py tutorial first
"""

# ============================================================================
# EXERCISE 1: Add a New State Field (BEGINNER)
# ============================================================================
print("=" * 70)
print("EXERCISE 1: Add a New State Field")
print("=" * 70)
print("""
TASK:
Add a new field to AgentState called 'compliment_count' that tracks
how many compliments have been given.

STEPS:
1. Add 'compliment_count: int' to the AgentState TypedDict
2. Initialize it to 0 in the initial_state
3. In generate_compliment(), increment it by 1
4. Print it in the final output

EXPECTED OUTPUT:
When you run the agent, you should see "Compliments given: 1" in the final state

HINT:
- Regular int fields get REPLACED (not concatenated like lists)
- To increment: return {"compliment_count": state["compliment_count"] + 1}

WHY THIS MATTERS:
Learn the difference between concatenating state (lists) and replacing state (primitives)
""")
print()


# ============================================================================
# EXERCISE 2: Add Conditional Branching (INTERMEDIATE)
# ============================================================================
print("=" * 70)
print("EXERCISE 2: Add Conditional Branching")
print("=" * 70)
print("""
TASK:
Add a new node called 'check_mood' that decides the next step based on mood.
- If mood is "excited", go to a new node "give_bonus_compliment"
- Otherwise, go directly to "summarize"

STEPS:
1. Create a new function: give_bonus_compliment(state)
   - Add an extra enthusiastic message to the messages list
2. Create a new function: check_mood(state)
   - This is a routing function (doesn't update state, just returns current state)
3. Replace the direct edge from "compliment" to "summarize"
4. Add: workflow.add_conditional_edges(
       "compliment",
       check_mood,
       {
           "excited": "bonus_compliment",
           "other": "summarize"
       }
   )

EXPECTED OUTPUT:
For coding/Python interests, you should see an extra bonus compliment

HINT:
- Conditional edges use a function that returns a STRING (the next node name)
- The routing function: return "excited" if state["mood"] == "excited" else "other"

WHY THIS MATTERS:
Real agents need to make decisions and branch based on state!
Graph-based workflows excel at this.
""")
print()


# ============================================================================
# EXERCISE 3: Add a Feedback Loop (INTERMEDIATE)
# ============================================================================
print("=" * 70)
print("EXERCISE 3: Add a Feedback Loop")
print("=" * 70)
print("""
TASK:
Modify the agent to ask for user feedback and potentially give another compliment.

STEPS:
1. Add 'feedback_received: bool' to AgentState (default: False)
2. Add 'feedback_positive: bool' to AgentState
3. Create a new node 'ask_feedback' that:
   - Adds message: "Was this compliment helpful? (simulating: yes)"
   - Sets feedback_received = True, feedback_positive = True
4. Create a routing function that:
   - If feedback_positive is True, go to "generate_compliment" again
   - Otherwise, go to "summarize"
5. Limit compliments to max 2 (add a counter to prevent infinite loop)

EXPECTED OUTPUT:
Agent should give 2 compliments total before summarizing

HINT:
- Use conditional_edges to create the loop
- Add a counter: 'compliment_count' to track iterations
- Exit when compliment_count >= 2

WHY THIS MATTERS:
Feedback loops are essential for interactive agents!
This is how chatbots handle multi-turn conversations.
""")
print()


# ============================================================================
# EXERCISE 4: Add Multiple Interests (ADVANCED)
# ============================================================================
print("=" * 70)
print("EXERCISE 4: Support Multiple Interests")
print("=" * 70)
print("""
TASK:
Modify the agent to handle a LIST of interests instead of a single interest.

STEPS:
1. Change 'user_interest: str' to 'user_interests: Annotated[list[str], operator.add]'
2. Modify analyze_interest() to loop through all interests
3. Generate a compliment for EACH interest
4. Update the initial state to accept a list: ["Python", "art", "music"]

EXPECTED OUTPUT:
Agent should generate separate analyses and compliments for each interest

HINT:
- Use a for loop in analyze_interest(): for interest in state["user_interests"]
- Each iteration should append to messages list
- Generate multiple compliments based on all interests

WHY THIS MATTERS:
Real-world agents need to handle complex, multi-faceted data.
List concatenation makes this elegant!
""")
print()


# ============================================================================
# EXERCISE 5: Add Error Handling (ADVANCED)
# ============================================================================
print("=" * 70)
print("EXERCISE 5: Add Error Handling")
print("=" * 70)
print("""
TASK:
Add validation and error handling to make the agent more robust.

STEPS:
1. Create a new node 'validate_input' at the start
2. Check if user_name is empty or user_interest is empty
3. If invalid, add an error message and go to END (skip other nodes)
4. If valid, proceed to 'greet' node
5. Use conditional_edges to route based on validation result

EXPECTED OUTPUT:
- Empty name/interest: Should print error and exit
- Valid input: Should work normally

HINT:
- Validation function returns "valid" or "invalid"
- Conditional edge maps these to node names or END
- Add 'error: str' field to state for error messages

WHY THIS MATTERS:
Production agents MUST handle bad input gracefully!
Never assume user input is perfect.
""")
print()


# ============================================================================
# EXERCISE 6: Integrate with Real AI (EXPERT)
# ============================================================================
print("=" * 70)
print("EXERCISE 6: Connect to Real AI Model")
print("=" * 70)
print("""
TASK:
Replace the hardcoded compliments with actual AI-generated compliments.

STEPS:
1. Install: pip install anthropic
2. Import: from anthropic import Anthropic
3. In generate_compliment(), use Claude API:

   client = Anthropic(api_key="your-api-key")
   message = client.messages.create(
       model="claude-3-5-sonnet-20241022",
       max_tokens=100,
       messages=[{
           "role": "user",
           "content": f"Give a sincere compliment to {name} who likes {interest}"
       }]
   )
   compliment = message.content[0].text

4. Handle API errors (try/except)
5. Test with real user input

EXPECTED OUTPUT:
Unique, AI-generated compliments each time you run!

HINT:
- Store API key in environment variable: os.getenv("ANTHROPIC_API_KEY")
- Add timeout and retry logic
- Fall back to hardcoded compliments if API fails

WHY THIS MATTERS:
This is where LangGraph shines - orchestrating real AI model calls!
You've built the infrastructure, now add real intelligence.
""")
print()


# ============================================================================
# BONUS CHALLENGE: Build Your Own Agent
# ============================================================================
print("=" * 70)
print("BONUS: Build Your Own Agent from Scratch")
print("=" * 70)
print("""
TASK:
Using what you've learned, build a completely new agent!

IDEAS:
1. Recipe Recommendation Agent
   - Input: dietary restrictions, available ingredients
   - State: restrictions (str), ingredients (list), recipe_steps (list)
   - Nodes: validate_ingredients → suggest_recipe → add_cooking_tips → summarize

2. Study Buddy Agent
   - Input: subject, difficulty level, time available
   - State: subject (str), flashcards (list), quiz_questions (list)
   - Nodes: generate_flashcards → create_quiz → provide_summary

3. Travel Planner Agent
   - Input: destination, budget, interests
   - State: destination (str), activities (list), budget_breakdown (list)
   - Nodes: suggest_activities → calculate_costs → create_itinerary

REQUIREMENTS:
- At least 3 nodes
- Use state concatenation (Annotated with operator.add)
- Include at least one conditional edge
- Print comprehensive final state

WHY THIS MATTERS:
Building from scratch solidifies your understanding!
You'll encounter edge cases and design decisions that teach you more than following tutorials.
""")
print()


# ============================================================================
# TESTING CHECKLIST
# ============================================================================
print("=" * 70)
print("Testing Checklist")
print("=" * 70)
print("""
After completing each exercise, verify:

□ Code runs without errors
□ State fields are properly typed in TypedDict
□ Messages are concatenating (not replacing)
□ Each node returns a dictionary with state updates
□ Graph edges connect in the right order
□ Final state contains all expected fields
□ Output is readable and makes sense

DEBUGGING TIPS:
- Print state at the START of each node function
- Use type hints to catch errors early
- Test with edge cases (empty strings, long lists, etc.)
- Check if messages list is growing (concatenation working)
- Verify final_state has all accumulated data
""")
print()


# ============================================================================
# LEARNING OUTCOMES
# ============================================================================
print("=" * 70)
print("What You'll Learn")
print("=" * 70)
print("""
By completing these exercises, you'll master:

✅ State Management
   - TypedDict for type-safe state structures
   - Annotated with operator.add for concatenation vs replacement
   - Accessing and updating nested state

✅ Graph Architecture
   - Designing node functions (single responsibility)
   - Connecting nodes with edges
   - Conditional branching and routing

✅ Workflow Orchestration
   - Sequential processing (linear flow)
   - Parallel processing (multiple branches)
   - Feedback loops (cyclic graphs)
   - Error handling and validation

✅ Modern Python Patterns
   - Type hints for better code quality
   - Function composition
   - Separation of concerns
   - Dependency injection

✅ AI Agent Design
   - Stateful vs stateless agents
   - Multi-turn conversations
   - Context accumulation
   - Decision-making logic

Ready to start? Open langgraph_compliment_agent.py and begin with Exercise 1!
""")
