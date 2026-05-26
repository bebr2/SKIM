
import random
def get_qa_message_not_react(skills, query, need_system_prompt=True):
    """
    skills: list of str
    """
    
    if need_system_prompt:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]
    else:
        messages = []
    question = "Relevant Skill:\n"
    skill_map = {}
    for i, skill in enumerate(skills):
        identifier = f"skill_{i}"
        if i == len(skills) - 1:
            question += f"<skill>{identifier}</skill>\n\n"
        else:
            question += f"<skill>{identifier}</skill>\n---\n"
        skill_map[identifier] = skill
    
    messages.append({
        "role": "user",
        "content": question + query
    })
        
    return {
        "messages": messages,
        "skill_map": skill_map
    }
    
def get_qa_tool_prompt_react(skills, query, tools):
    """
    skills: list of str
    """
    question = "Relevant Skill:\n"
    skill_map = {}
    for i, skill in enumerate(skills):
        identifier = f"skill_{i}"
        if i == len(skills) - 1:
            question += f"<skill>{identifier}</skill>\n\n"
        else:
            question += f"<skill>{identifier}</skill>\n---\n"
        skill_map[identifier] = skill
        
    old_tools = [
        "Calculate[formula]: Calculates the formula and returns the result.",
        "RetrieveAgenda[keyword]: Retrieves the agenda related to keyword.",
        "LoadDB[DBName]: Loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.",
        "FilterDB[condition]: Filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.",
        "GetValue[column_name]: Returns the value of the column column_name in the database DBName.",
        "LoadGraph[GraphName]: Loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.",
        "NeighbourCheck[GraphName, Node]: Lists the neighbours of the node Node in the graph GraphName and returns the neighbours.",
        "NodeCheck[GraphName, Node]: Returns the detailed attribute information of Node.",
        "EdgeCheck[GraphName, Node1, Node2]: Returns the detailed attribute information of the edge between Node1 and Node2.",
        "Finish[answer]: Returns the answer and finishes the task."
    ]
    for tool in tools:
        tool_name = tool["name"]
        tool_desc = tool["description"]
        tool_str = f"{tool_name}["
        for j, param in enumerate(tool["input_params"]):
            if j != len(tool["input_params"]) - 1:
                tool_str += f"{param['name']},"
            else:
                tool_str += f"{param['name']}]"
        tool_str += f": {tool_desc} "
        for param in tool["input_params"]:
            tool_str += f"The parameter {param['name']} is {param['description'].lower()}. "
        old_tools.append(tool_str)
    random.shuffle(old_tools)
    system_instruction = (
        "Solve a question answering task with interleaving Thought, Action, Observation steps. "
        f"Thought can reason about the current situation, and Action can be {len(tools)+10} types:\n"
        + "\n".join([f"({idx+1}) {tool_str}" for idx, tool_str in enumerate(old_tools)]) + "\n"
        "You may take as many steps as necessary."
    )

    examples = (
        "Here are some examples:\n"
        "Question: How much longer is the air time of flight DL123 from JFK to LAX compared to flight UA456 from SFO to ORD on 2023-05-10?\n"
        "Thought 1: This is a question related to flights. We need to load the flights database.\n"
        "Action 1: LoadDB[flights]\n"
        "Observation 1: We have successfully loaded the flights database, including the following columns: FlightDate, Airline, Origin, Dest, Flight_Number_Operating_Airline, AirTime...\n"
        "Thought 2: We need to filter the information for flight DL123.\n"
        "Action 2: FilterDB[Flight_Number_Operating_Airline=123, FlightDate=2023-05-10, Origin=JFK, Dest=LAX]\n"
        "Observation 2: We have successfully filtered the data (1 row).\n"
        "Thought 3: We then need to know its air time.\n"
        "Action 3: GetValue[AirTime]\n"
        "Observation 3: 340.0\n"
        "Thought 4: We need to filter the information for flight UA456.\n"
        "Action 4: FilterDB[Flight_Number_Operating_Airline=456, FlightDate=2023-05-10, Origin=SFO, Dest=ORD]\n"
        "Observation 4: We have successfully filtered the data (1 row).\n"
        "Thought 5: We then need to know its air time.\n"
        "Action 5: GetValue[AirTime]\n"
        "Observation 5: 220.0\n"
        "Thought 6: To compute the difference, we subtract UA456's air time from DL123's air time.\n"
        "Action 6: Calculate[340.0-220.0]\n"
        "Observation 6: 120.0\n"
        "Thought 7: After calculation, the answer is 120.\n"
        "Action 7: Finish[120]\n"
        "\n"
        "Question: What time does Alice's dentist appointment start?\n"
        "Thought 1: The question is asking for the start time of Alice's dentist appointment from the agenda.\n"
        "Action 1: RetrieveAgenda[Alice's dentist appointment]\n"
        "Observation 1: On March 15, 2023, Alice has a dentist appointment at Smile Clinic. The appointment starts at 10:30 AM and is expected to last for an hour.\n"
        "Thought 2: The appointment starts at 10:30 AM.\n"
        "Action 2: Finish[10:30 AM]\n"
        "\n"
        "Question: In what year was the paper Attention Is All You Need published according to DBLP?\n"
        "Thought 1: The question is asking some basic information of a dblp paper. We need to load the DBLP graph.\n"
        "Action 1: LoadGraph[dblp]\n"
        "Observation 1: DBLP graph is loaded.\n"
        "Thought 2: The question is asking the published date of a paper, we need to check the node from the PaperNet in DBLP graph.\n"
        "Action 2: NodeCheck[PaperNet, Attention Is All You Need]\n"
        "Observation 2: {'title': 'Attention Is All You Need', 'year': 2017, 'venue': {'raw': 'NIPS'}, 'n_citation': 40000, 'keywords': [], 'doc_type': 'Conference'}\n"
        "Thought 3: The published date of the paper is 2017.\n"
        "Action 3: Finish[2017]\n"
        "\n"
        "Question: If a store sells 15 apples a day, how many apples does it sell in 4 weeks?\n"
        "Thought 1: We need to calculate the total number of apples sold in 4 weeks. There are 7 days in a week.\n"
        "Action 1: Calculate[15*7*4]\n"
        "Observation 1: 420\n"
        "Thought 2: The store sells 420 apples in 4 weeks.\n"
        "Action 2: Finish[420]\n"
        "\n"
        "Question: How many common papers do Yoshua Bengio and Yann LeCun have in the DBLP graph?\n"
        "Thought 1: The question is asking the number of common papers of two persons, we need to load the DBLP graph.\n"
        "Action 1: LoadGraph[dblp]\n"
        "Observation 1: DBLP graph is loaded.\n"
        "Thought 2: We need to check the edge between Yoshua Bengio and Yann LeCun in the AuthorNet to find their common papers.\n"
        "Action 2: EdgeCheck[AuthorNet, Yoshua Bengio, Yann LeCun]\n"
        "Observation 2: {'weight': 2, 'papers': ['Deep Learning', 'Gradient-Based Learning Applied to Document Recognition'], 'n_citation': [35000, 40000]}\n"
        "Thought 3: The list of common papers contains 2 items.\n"
        "Action 3: Finish[2]\n"
        "\n"
        "(END OF EXAMPLES)\n"
    )
    
    user_prompt = examples + question + f"Question: {query}\n" + "Thought 1:"
    return {
        "system_instruction": system_instruction,
        "user_prompt": user_prompt,
        "skill_map": skill_map
    }

