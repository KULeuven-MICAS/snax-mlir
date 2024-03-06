import json
import pandas as pd
# Open the test_results.json file
with open('test_results.json', 'r') as file:
    data = json.load(file)

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data, columns=['layout', 'backend', 'size', 'success', 'cycles'])

# order the data by layout, backend, and size
df = df.sort_values(by=['layout', 'backend', 'size'])

print(df)


