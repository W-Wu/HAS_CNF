# Human Annotator Simulation (HAS) via CNFs
Code for ''It HAS to be Subjective: Human Annotator Simulation via Zero-shot Density Estimation''

## Dependencies
PyTorch==1.11.1  
speechbrain==0.5.14  
normflows==1.6  
numpy==1.21.0  
scikit-learn==1.0.2  
statsmodels==0.13.5  

## Data preparation
### Emotion classification on [MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
Prepare label: `python3 data_preparation/prep_msp-label.py`  
Prepare training scp: `python3 data_preparation/prep_msp-scp.py`
### Hate speech detection on [HateXplain](https://cdn.aaai.org/ojs/17745/17745-13-21239-1-2-20210518.pdf)
`python3 data_preparation/prep_hx.py`
### Speech quality assessment on [SOMOS](https://www.isca-speech.org/archive/interspeech_2022/maniati22_interspeech.html)
`python3 data_preparation/prep_somos.py`

## Training
### Conditional softmax flow (S-CNF) for Categorical Annotations
`python3 Train_S-CNF.py Train_S-CNF.yaml --output_folder='exp'`

### Conditional Integer Flows (I-CNF) for Ordinal Annotations
`python3 Train_I-CNF.py Train_I-CNF.yaml --output_folder='exp'`

## Scoring
For S-CNF: `python3 scoring_S-CNF.py exp/test_outcome-E{PLACEHOLDER}.npy`  
For I-CNF: `python3 scoring_I-CNF.py exp/test_outcome-E{PLACEHOLDER}.npy`  
