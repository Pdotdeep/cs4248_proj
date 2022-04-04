import json
import os
import pandas as pd
from rouge import Rouge
import statistics
import sys

def evaluate(data):

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

  droptext = lambda flag: results.drop("Text", axis=1) if flag else None
  droptext(True)
  results.to_csv(f"../output/scored_{name}.csv")

  # Calculate macroscopic performance
  scores_averaged = rouge.get_scores(generated, actual, avg=True)
  print()
  print("#####################################")
  print(f"Using predictions: {name}")
  print()
  print(f"Average number of violations according to gingerit: {statistics.mean(violations)}")
  print()
  print("ROUGE score")
  print("-----------")
  print(scores_averaged)
  print()

  return results

if len(sys.argv) > 1:
  generated_path = [sys.argv[1]]
else:
  paths = os.listdir("../output")
  generated_path = [path for path in paths if path.startswith("generated")]

for path in generated_path:
  full_path = os.path.join("../output", path)
  # Load generated text
  with open(full_path) as f:
    data = json.load(f)

  # Write individual scores to file
  name = path.split("/")[-1].split(".")[0]
  results = evaluate(data)
