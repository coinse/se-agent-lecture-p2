

def process_task_instruction(instruction: str, target_url: str):
    return {
        "role": "user",
        "content": f"""Perform the following task:
{instruction}

You must perform the task inside the target webpage: {target_url}.
If you accidentally navigate away from the page, try to return to it.

After completing the task, please provide a brief testing report about the actions you took, report any issues you encountered."""
    }
