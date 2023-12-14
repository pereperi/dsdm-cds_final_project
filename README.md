This repository called "dsdm-cds_final_project" is our repository for our Final Project of Computing for Data Science in BSE. 

***Members:*** Álvaro Ortiz, Sebastien Boxho, Pere Pericot, Guillem Mirabent Rubinat.

The objective of this project is to build the architecture of a library that is scalable and functional for a Data Science project.

***Data used:***
FIFA 2022 player database with a comprehensive roster of variables available for analysis. 
The primary objective of this undertaking is to predict and forecast the player's designated position based on these aforementioned variables.

id: Unique identifier for each player.

short_name: Short name or nickname of the player.

overall: Player's overall rating, representing their overall skill level.

potential: Player's potential rating, indicating their potential skill growth.

value_eur: Player's market value in euros.

wage_eur: Player's weekly wage in euros.

birthday_date: Player's date of birth.

height_cm: Player's height in centimeters.

weight_kg: Player's weight in kilograms.

club_name: Name of the player's club.

league_name: Name of the league the club belongs to.

position: Player's preferred playing position.

preferred_foot: Player's preferred kicking foot (left or right).

weak_foot: Player's weak foot rating, indicating their weaker kicking foot's ability.

skill_moves: Player's skill moves rating, representing their dribbling and ball control skills.

international_reputation: Player's international reputation level.

pace, shooting, passing, dribbling, defending, physic: Attributes representing different aspects of a player's playing style and skills.

mentality_aggression, mentality_vision, mentality_composure: Attributes representing mental aspects of a player's game.

attacking_crossing, attacking_finishing, attacking_heading_accuracy: Attributes related to attacking and finishing skills.

movement_acceleration, movement_sprint_speed, movement_agility: Attributes related to a player's speed and agility.

power_shot_power, power_jumping, power_stamina: Attributes representing a player's physical power and endurance.

defending_marking_awareness, defending_standing_tackle, defending_sliding_tackle: Attributes representing a player's defensive skills.

goalkeeping_diving, goalkeeping_handling, goalkeeping_positioning: Goalkeeping attributes related to diving, handling, and positioning.

goalkeeping_reflexes, goalkeeping_speed: Attributes representing a goalkeeper's reflexes and speed.

***Library Architecture:***
<pre>
dsdm-cds_final_project/
│
├── data/                           # Data directory
│   └── train.csv                   # Train dataset
│
├── mylib/                          # Main library package
|   ├── __init__.py
|   └── position_prediction/        # Library for position prediction
│       ├── __init__.py             # Initializes the mylib package
│       ├── features.py             # Module for feature related functions
│       ├── model.py                # Module for model related functions
│       ├── pipeline.py             # Module for pipeline functions
│       ├── preprocessing.py        # Module for preprocessing functions
│       └── utils.py                # Utility functions
│
├── tests/                          # Test cases
│   ├── test_feature_engineering.py # Test for feature engineering
│   └── test_preprocessing.py       # Test for preprocessing
│
├── pipeline_caller.ipynb           # Main pipeline caller notebook
├── README.md                       # README file
├── requirements.txt                # List of dependencies
└── setup.py                        # Setup script for the library
</pre>


***Guidelines for newcomers to add elements to the library:***
0. Basics
- Create a branch of the repository as it is to work on it.
- Document relevant information in an additional/new .txt file which must be created inside the directory '/documentation' and must as well be appropriately named.
- Any relevant addition/modification for the following sections must be properly documented and justified in its own .txt file as described.
- You can use the notebook 'pipeline_caller.ipynb' to test your advances, but don't merge it with the main unless relevant explanations on the markdown sections are added to it.
- Please do merge the rest of the files when your additions are done.

1. New preprocessors
- Preprocessors should accept and return Pandas DataFrames or Series.
- Document any data format assumptions and required library versions.
- Implement error handling for data inconsistencies or format issues.
- Write comprehensive unit tests for the added preprocessors, covering various data scenarios.

2. Features
- Document any external data or resources used in feature calculation.
- Provide unit tests validating the new feature extraction.

3. Models
- Ensure models are compatible with the existing data preprocessing and feature engineering pipeline.
- Follow a consistent structure for new/alternate models. Each model should have its own function and its own 'elif' in the pipeline last 'for' loop.
- Document the model's purpose and its mathematical rationale and explanation.
- Add the relevant hyperparameters to 'hyperparameters_GridSearch' in model.py.
- The model must be contained in its own function, where it's trained. To try different models, separately or at once, make sure to use the notebook.
- Follow the same isolation in function for the different metrics.

4. Metrics
- Metrics should be implemented within their designed function in 'pipeline.py', 'evaluate_model'.
- Document the mathematical background and interpretation for each new metric.

5. Additionals
- Should you need to add some functionalities not established anywhere else, please use the file 'utils.py'.
- Make sure to properly implement the added 'utils' to the pipline and to make sure that the call from the notebook works correctly afterwards.
