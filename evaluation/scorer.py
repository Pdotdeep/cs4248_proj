import json
import pandas as pd
from rouge import Rouge
import statistics
import sys

generated_path = sys.argv[1]

# Load generated text
with open(generated_path) as f:
  data = json.load(f)

text = list(data["Text"].values())
generated = list(data["Generated"].values())
actual = list(data["Actual"].values())

# Rouge scoring
rouge = Rouge()
scores = rouge.get_scores(generated, actual, avg=False)
results = pd.DataFrame(scores)
results.loc[:, "Generated"] = generated
results.loc[:, "Actual"] = actual
results.loc[:, "Text"] = text

# Gingerit
import gingerit
from gingerit.gingerit import GingerIt
parser = GingerIt()
parsedicts = [parser.parse(sentence) for sentence in generated]
matches = [parsedict['corrections'] for parsedict in parsedicts]
violations = [len(matched) for matched in matches]
results.loc[:, "Matches"] = matches
results.loc[:, "Violations"] = violations

# Write individual scores to file
name = generated_path.split("/")[-1].split(".")[0]
droptext = lambda flag: results.drop("Text", axis=1) if flag else None
droptext(True)
results.to_csv(f"../output/scored_{name}.csv")

# Calculate macroscopic performance
scores_averaged = rouge.get_scores(generated, actual, avg=True)
print()
print(f"Average number of violations according to gingerit: {statistics.mean(violations)}")
print()
print("ROUGE score")
print("#####################################")
print(scores_averaged)
