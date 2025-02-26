import chz
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.asyncio_utils import generator_with_cleanup
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    JupyterComputerInterface,
)
from nanoeval.solvers.computer_tasks.solver import PythonCodingEval, strip_all_metadata
from nanoeval.solvers.computer_tasks.steps import (
    FinalResult,
    FinalResultSuccessful,
    FinalResultWithException,
)
from nanoeval.solvers.computer_tasks.task import ComputerTask, Grade
from typing_extensions import override
from nanoeval.solvers.computer_tasks.steps import FinalResultWithException, Step
from alcatraz.clusters.local import LocalConfig
import shlex

import asyncio
import functools
import os
import re
import subprocess
import threading
import traceback
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any, AsyncGenerator, ContextManager, Generator, Generic, TypeVar, cast

from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, AsyncGenerator, Generator
from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface

from openai import OpenAI
import os
import tiktoken


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)


def count_tokens(messages: list[dict[str, Any]], model: str = "gpt-4") -> int:
    """Count the number of tokens in a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    
    for message in messages:
        # Every message follows format: {"role": role, "content": content}
        num_tokens += 4  # Every message follows format: <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    
    return num_tokens

def trim_messages(messages: list[dict[str, Any]], max_tokens: int, model: str = "gpt-4") -> list[dict[str, Any]]:
    """Trim messages to fit within token limit by removing older messages."""
    while len(messages) > 1 and count_tokens(messages, model) > max_tokens:
        messages.pop(1)
    return messages

def get_model_response(messages: list[dict[str, Any]], model: str = "gpt-4o") -> str:
    messages = trim_messages(messages, 110000)
    
    chat_completion = client.chat.completions.create(
        messages=messages, # type: ignore
        model=model,
    )
    return chat_completion.choices[0].message.content # type: ignore

# Modified function to track token usage
def get_model_response_with_usage(messages: list[dict[str, Any]], model: str = "gpt-4o") -> tuple[str, dict]:
    """Get model response and track token usage."""
    messages = trim_messages(messages, 110000)
    
    chat_completion = client.chat.completions.create(
        messages=messages, # type: ignore
        model=model,
    )
    
    # Extract usage statistics
    usage = {
        "prompt_tokens": chat_completion.usage.prompt_tokens,
        "completion_tokens": chat_completion.usage.completion_tokens,
        "total_tokens": chat_completion.usage.total_tokens
    }
    
    return chat_completion.choices[0].message.content, usage


@chz.chz
class SimpleAgentSolver(PythonCodingSolver):
    name: str = "SimpleAgentSolver"
    model: str = "gpt-4o"
    token_usage_log_dir: str = "token_usage_logs"
    
    @chz.init_property
    def token_usage_log_file(self) -> str:
        # Create a timestamped log file name
        from datetime import datetime
        import os
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.token_usage_log_dir, exist_ok=True)
        
        # Set the log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            self.token_usage_log_dir, 
            f"token_usage_{timestamp}.jsonl"
        )
        print(f"Token usage will be logged to: {log_file}")
        return log_file

    def shortname(self) -> str:
        return "simple-solver"

    @asynccontextmanager
    async def _start_computer(self, task: ComputerTask) -> AsyncGenerator[ComputerInterface, None]:
        # replace with LocalCluster semantics

        alcatraz_env = task_to_alcatraz_config(task, LocalConfig(pull_from_registry=False))

        async with alcatraz_env.build() as cluster:
            yield AlcatrazComputerInterface(cluster_value=cluster)
    
    def _log_token_usage(self, task_id: str, usage_data: dict) -> None:
        """Log token usage data to a JSONL file."""
        import json
        from datetime import datetime
        import os
        
        # Create a log entry with timestamp, task ID, and usage data
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "model": self.model,
            "usage": {
                "prompt_tokens": usage_data["prompt_tokens"],
                "completion_tokens": usage_data["completion_tokens"],
                "total_tokens": usage_data["total_tokens"],
                "api_call_count": usage_data["api_call_count"]
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.token_usage_log_file), exist_ok=True)
        
        # Append to the JSONL file
        with open(self.token_usage_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        try:
            # Initialize token usage tracking for this task
            task_token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_call_count": 0
            }
            
            async with self._start_computer(task) as computer:
                print(computer)
                # 1. Run the task setup
                await task.setup(computer)

                # 2. Query the API / some agent
                messages = [] 
                for prompt_message in task.prompt:
                    messages.append({
                        "role": "user",
                        "content": str(prompt_message["content"]) # type: ignore
                    })
                messages.append({"role": "user", "content": """The repository is cloned in your CWD. You must send Python code in backticks in each response to me, and I will execute the code and send you back the result, for example:
                                                                       
```python
import os
print(os.listdir())
```

You can include other content in your message, as long as it has the Python code block. You must only submit one code block per message.
                                 
The python code block is your means of interacting with the environment; write Python code to view files, edit files, and run commands. Do not leave any steps incomplete. You must implement all fixes yourself.
                                                                       
When you are done, send "DONE" (and nothing else - you don't need a python code block), and we will grade your submission.
                                 
Please note that the Python code is not a Jupyter notebook; you must write a full Python script each time, including imports, and printing things if you want to see the output.
"""})
                max_turns = 30
                print(messages)

                for remaining_turns in range(max_turns, 0, -1):
                    model_response, usage = get_model_response_with_usage(messages, self.model)
                    print(model_response)
                    
                    # Update token usage for this task
                    task_token_usage["prompt_tokens"] += usage["prompt_tokens"]
                    task_token_usage["completion_tokens"] += usage["completion_tokens"]
                    task_token_usage["total_tokens"] += usage["total_tokens"]
                    task_token_usage["api_call_count"] += 1  # Increment API call counter
                    
                    messages.append({"role": "assistant", "content": model_response})

                    execution_output = None

                    #Check for user-tool calls
                    if "<user-tool>" in model_response:
                        print('User tool called.')
                        await computer.check_shell_command("unzip -q -P 'secretpasswordhere' /app/tests.zip -d /app/tests")
                        execution_output = await computer.check_shell_command("bash -c -i 'user-tool'")
                        print('User tool ran, with output:')
                        print(execution_output)
                        execution_output = 'User tool output is in /app/expensify/user_tool'
                        await computer.send_shell_command(
                            "rm -rf /app/tests"
                        )
                        await computer.send_shell_command("rm -rf /app/tests")
                    else:
                        # Extract Python code from the model response
                        python_blocks = re.findall(r"```python\s*\n(.*?)\n```", model_response, re.DOTALL)

                        if not python_blocks:
                            messages.append({
                                "role": "user",
                                "content": "Warning: No Python code blocks were found in the response. Please include Python code in your replies."
                            })
                            continue

                        code = dedent(python_blocks[0])

                        execution_output = await computer.send_shell_command(f"python -c {shlex.quote(code)}")
                        execution_output = execution_output.output.decode("utf-8", errors="replace")
                        
                    if model_response.lower() == "done":
                        print("Breaking because model is done!")
                        break


                    print(execution_output)

                    # Append the code and its output to the messages
                    messages.append({
                        "role": "user",
                        "content": f"{execution_output}\nTurns left: {remaining_turns - 1}"
                    })
                
                # Log the total token usage for this task
                self._log_token_usage(task.question_id, task_token_usage)
                print(f"Token usage for task {task.question_id}: {task_token_usage}")

                # 3. Grade and yield the final result
                grade = await task.grade(computer)
                yield FinalResultSuccessful(grade=grade)
        except Exception as e:
            print(f"Error: {e}")
            raise
            yield FinalResultSuccessful(
                grade=Grade(score=0, grader_log=f"Grading failed with error: {str(e)}")
            )