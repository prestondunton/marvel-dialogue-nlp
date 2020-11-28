# marvel-dialogue-nlp
A machine learning project that will use Natural Language Processing (NLP) to classify who says a line of dialogue

<p align="center">
  <img src="https://blog.umhb.edu/wp-content/uploads/2019/06/mcu-1920x1080.jpg" alt="MCU Banner" width="60%" height="60%">
</p>

## About the Project
With over 21 different movies, all spanning the same cinematic universe, Marvel movies are an interesting creation of character, dialogue, and plot.  A key defining feature of these movies is the large amount of characters represented and developed.  I wanted to explore NLP, and figured that exploring the dialogue in these 
movies would be a very fun thing to do.  The goal is to predict the character's name given their line.  For example, given the line “I am inevitable,”  we would want to 
predict “Thanos” because Thanos is the character that says that line.  

This project is still in progress, but when finished, I'll add a section about which model performed the best and its results.


## About the Dataset
This repository contains a newly created dataset to train and test models on, as well as several Jupyter Notebooks that describe the process used to create each `.csv`.  The end file, [mcu.csv](https://github.com/prestondunton/marvel-dialogue-nlp/data/mcu.csv), contains columns `character` and `line` that hold the dialogue for several movies from the MCU. There are more columns that provide additional features for use.  See [/data/MCU.ipynb](https://github.com/prestondunton/marvel-dialogue-nlp/data/MCU.ipynb) for more details on those features. For individual movies, the corresponding `.csv` can be found in [/data/cleaned/](https://github.com/prestondunton/marvel-dialogue-nlp/data/cleaned) and contain columns `character` and `line`.  Each movie file was created using the same process, though improvements were found as more movies were processed.

The scripts `.pdf`s were obtained from [Script Slug](https://www.scriptslug.com/scripts/category/marvel), though other copies of the Marvel released scripts can be found online elsewhere.  Not all of the MCU movie scripts were released, so this dataset only contains a subset of the movies in the MCU.  Transcripts exist for all 21 movies, though these transcripts can contain many errors, so they were not used.  Additionally, creating each `.csv` takes quite a bit of time, so currently, this dataset only contains 5 movies (listed below).  Hopefully, I'll find time to add more movies.

If you spot a mistake, please let me know so I can correct it!


| Movies Included                       |
| ------------------------------------- |
| Iron Man (2008)                       |
| The Avengers (2012)                   |
| Thor: Ragnorak (2017)                 |
| Guardians of the Galaxy Vol. 2 (2017) |
| Avengers Endgame (2019)               |
