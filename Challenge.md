# Guidelines for the SoccerNet MV-Foul Challenge

The 2nd [SoccerNet MV-Foul challenge]() will be held from October to June 2025!
Subscribe (watch) the repo to receive the latest info regarding timeline and prizes!

SoccerNet MV-Foul is a large-scale dataset that benchmarks the tasks of a football referee, identifying fouls, determining yellow and red cards and the type of foul.
For the MV-Foul challenge, participants have to determine from multi-view clips two labels. The first labels corresponds to if it is a foul or not, i.e. No Offence, Offence + No Card, Offence + Yellow Card, Offence + Red Card, while the second label determine the type of foul, i.e. Standing Tackle, Tackle, Holding, Pushing, Challenge, Dive, High Leg, Elbowing.
SoccerNet MV-Foul is composed of 3k annotations, span 500 complete soccer games from six main European leagues, covering three seasons from 2014 to 2017.

We provide an [evaluation server](www.google.com) for anyone competing in the SoccerNet MV-Foul. 
This evaluation server handles predictions for the open **test** set and the segregated **challenge** set.

Winners will be announced during the CVSports workshop at CVPR in June 2025. 
Prizes 💲💲💲 include $1000 cash award, sponsored by XXXX.

## Youtube video

Check out our video tutorial on the MV-Foul challenge!

[![IMAGE ALT TEXT HERE](images/Thumbnail.jpg)](https://youtu.be/Ir-6D3j_lkA?si=Uwni8jngdsDQrf6w)

## Who can participate / How to participate?

 - Any individual can participate in the challenge, except the organizers.
 - The participants are recommended to form a team to participate.
 - Each team can have one or more members. 
 - An individual/team can compete on both task.
 - An individual associated with multiple teams (for a given task) or a team with multiple accounts will be disqualified.
 - Participants can use any **public** dataset to pretrain their model. Any public dataset or codebase used for the challenge must be mentioned in the final report.
 - Participants are allowed to train their final model on all provided data (train + valid + test sets) before evaluating on the challenge set.
 - If you have any doubts regarding these rules, please contact the challenge administrators.

## How to win / What is the prize?

 - The winner will the individual/team who reaches the highest **balanced accuracy** performance on the **challenge** set.
 - The deadline to submit your results is May 30th at 11.59 pm  Pacific Time.
 - The teams that perform best in each task will be granted $1000 from our sponsor XXX.
 - In order to be eligible for the prize, we require the individual/team to provide a short report describing the details of the methodology (CVPR format, max 3 pages)


## Important dates

Note that these dates are tentative and subject to change if necessary.

 - **October 30:** Launch of SoccerNet challenges.
 - **Mid-March:** Deadline for CVsports. We encourage participants to publish their results at the CVsports workshop! 
 - **April 30:** Deadline for the SoccerNet Challenges. The participants have to submit a report.
 - **June TBD:** A full-day workshop at CVPR 2025.


End of October: Launch of SoccerNet challenges.
Mid-March: Deadline for CVsports. We will improve the communication and invite every participant to submit a paper. The review process is quick and we will have the list of accepted paper by mid-April
End of April: Deadline for the SoccerNet Challenges. The participants will have to provide a report, they can refer to their CVsports paper.
May: We review the reports and confirm the leaderboard. We prepare presentation material for CVsports
June: CVsports @CVPR: We release the results of the SoccerNet Challenges 2025.
Summer: We publish on arxiv the usual SoccerNet challenge results paper.


## Contact

For any further doubt or concern, please raise a GitHub issue in this repository, or contact us directly on [Discord](https://discord.gg/SM8uHj9mkP).
