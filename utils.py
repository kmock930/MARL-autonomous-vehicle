# Modify the `new_pos` function to check roles using the Agent class
def new_pos(agent_position: tuple[int, int], action: ACTION_SPACE, agents: list):
    x, y = agent_position
    dx, dy = action.value

    new_pos = (x + dx, y + dy)

    # Check if the new position is within the grid
    if not (0 <= new_pos[0] < env.env_configurations["rowSize"] and 0 <= new_pos[1] < env.env_configurations["colSize"]):
        print("Out of Bounds")  # Debugging message
        return agent_position  # Reverse because Invalid Move

    # Check if the new position is occupied by another agent
    for agent in agents:
        if agent["position"] == new_pos:
            print(f"Agent at {agent_position} collided with agent at {new_pos}")
            return agent_position  # Reverse because Invalid Move


    # Check if the new position is a hard obstacle
    if env.obstacles[new_pos[0], new_pos[1]] in [OBSTACLE_HARD]:
        print("Obstacle Collision")  # Debugging message
        return agent_position  # Reverse because Invalid Move

    print("Valid Move")  # Debugging message
    return new_pos