"""Few-shot examples for ToolQA ReAct agent.

This file intentionally keeps a compact fallback set.
When --skillrag_root is provided, the runner tries to load the full
ToolQA few-shots from the external SkillRAG repository first.
"""

TOOLQA_EXAMPLES = """Question: When was the paper Learning the Principle of Least Action with Reinforcement Learning. published?
Thought 1: The question asks for publication year in DBLP, so I should load the graph first.
Action 1: LoadGraph[dblp]
Observation 1: DBLP data is loaded, including two graphs: AuthorNet and PaperNet.
Thought 2: Now I should check the target paper node details.
Action 2: NodeCheck[PaperNet, Learning the Principle of Least Action with Reinforcement Learning.]
Observation 2: {'title': 'Learning the Principle of Least Action with Reinforcement Learning.', 'year': 2021}
Thought 3: I have the publication year.
Action 3: Finish[2021]

Question: What was the trading volume of coffee on 2000-01-14?
Thought 1: I can use SQL over the coffee table.
Action 1: SQLInterpreter[SELECT Volume FROM coffee.coffee_data WHERE Date = '2000-01-14']
Observation 1: Volume: 10115
Thought 2: The answer is ready.
Action 2: Finish[10115]
"""
