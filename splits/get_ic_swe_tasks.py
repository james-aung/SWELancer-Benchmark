import pandas as pd

# Read the tasks CSV
tasks = pd.read_csv('swelancer_tasks.csv')

# Filter for ic_swe variant
ic_swe_tasks = tasks[tasks['variant'] == 'ic_swe']['question_id'].tolist()

print(f"Found {len(ic_swe_tasks)} 'ic_swe' tasks")

with open('ic_swe_task_ids.txt', 'w') as f:
    f.write(' '.join(map(str, ic_swe_tasks)))

print("Task IDs saved to ic_swe_task_ids.txt")