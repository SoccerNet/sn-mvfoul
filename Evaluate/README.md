# Evaluation

We provide evaluation functions directly integrated in our pip package (pip install SoccerNet) as well as an evaluation server on EvalAI.

We defined the following metric for MV-Foul recognition. You have to predict two labels:

1. The first label determines if it is a foul or not and the corresponding severity. We have 4 classes: No Offence, Offence + No card, Offence + Yellow card and Offence + Red card.
    We use the balanced accuracy as metric.
2. The second label determines the type of action. We have 8 classes: Tackling, Standing tackling, High leg, Holding, Pushing, Elbowing, Challenge and Dive.
   We use the balanced accuracy as metric.

For the leaderboard we take the mean of the two balanced accuracies.

# Output format

To submit your results on EvalAI or to use the integreted function of the pip package, the predictions of the network have to be saved in a specific format with a json file constructed as follows:

```
{
    "Actions": {
        "0": {
            "Action class": "High leg",
            "Offence": "Offence",
            "Severity": "3.0"
        },
        "1": {
            "Action class": "Standing tackling",
            "Offence": "Offence",
            "Severity": "1.0"
        },
        "2": {
            "Action class": "Challenge",
            "Offence": "No offence",
            "Severity": ""
        },
        "3": {
            "Action class": "Tackling",
            "Offence": "Offence",
            "Severity": "3.0"
        },
        "5": {
            "Action class": "Pushing",
            "Offence": "Offence",
            "Severity": "5.0"
        },
        ....
}

```

# Evaluation on the Chall set on EvalAI

In order to participate in the challenge, the predictions for the challenge set should be submitted as a JSON file on the online evaluation platform EvalAI. Further information can be found on the EvalAI website.

