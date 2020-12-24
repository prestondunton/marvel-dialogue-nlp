# marvel-dialogue-nlp
A machine learning project that will use Natural Language Processing (NLP) to classify who says a line of dialogue

<p align="center">
  <img src="https://blog.umhb.edu/wp-content/uploads/2019/06/mcu-1920x1080.jpg" alt="MCU Banner" width="60%" height="60%">
</p>

## About the Project
With over 21 different movies, all spanning the same cinematic universe, Marvel movies are an interesting creation of character, dialogue, and plot.  A key defining feature of these movies is the large amount of characters represented and developed.  I wanted to explore NLP, and figured that exploring the dialogue in these 
movies would be a very fun thing to do.  The problem whished to be accomplished in this project is an NLP classification problem.  The goal was to create a model that can predict a character's name given a line of their dialogue from a Marvel Cinematic Universe (MCU) movie.  Data was taken from Marvel released scripts and transformed into labels of names and feature documents of their dialogue.

## Results
A summary of the results of the project can be read in [Final Report.ipynb](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/Final%20Report.ipynb).  I trained 12 different models, each a different combination of using stemming, using TF / IDF, and a different classifier (Naive Bayes, Random Forest, SVM).  A Naive Bayes classifier with word stemming and TF / IDF transformations gave the best results, which were around 29% balanced accuracy.  This is shown to be so low because the model needs more data to train from.  If I have time, I'll revisit this by adding more movies to the dataset and retrain the models.


## About the Dataset
This repository contains a newly created dataset to train and test models on, as well as several Jupyter Notebooks that describe the process used to create each `.csv`.  These Jupyter notebooks explain the process of parsing the `.pdf`s with the `pandas` library.  The end file, [mcu.csv](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/mcu.csv), contains columns `character` and `line` that hold the dialogue for several movies from the MCU. There are more columns that provide additional features for context, but were not used in this project.  See [/data/MCU.ipynb](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/MCU.ipynb) for more details on those features. For individual movies, the corresponding `.csv` can be found in [/data/cleaned/](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned) and contain columns `character` and `line`.  Each movie file was created using the same partially automated process, though improvements were found as more movies were processed.

This dataset uses a combination of original scripts and transcripts from the MCU movies.  The original script `.pdf`s were obtained from [Script Slug](https://www.scriptslug.com/scripts/category/marvel), though other copies of the Marvel released scripts can be found online elsewhere.  Transcripts were taken from [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Category:Marvel_Transcripts). Transcripts were copied and pasted into `.txt` files, and then processed using `pandas`. Creating each `.csv` took quite a bit of time, so currently, this dataset only contains 6 movies.  The table below contains metadata about the dataset and each movie's source.  This table is stored in [movies.csv](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/movies.csv).

If you spot a mistake in the dataset, please let me know so I can correct it.

| Movie                          | Year | Is Transcript | Lines | Source Link |
| ------------------------------ | ---- | ------------- | ----- | ---------- |
| Iron Man                       | 2008 | ❌            | 834  | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/iron-man-2008.pdf) |
| The Avengers                   | 2012 | ❌            | 1027 | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/the-avengers-2012.pdf) |
| Thor: Ragnorak                 | 2017 | ❌            | 961  | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/thor-ragnorak-2017.pdf) |
| Guardians of the Galaxy Vol. 2 | 2017 | ❌            | 993  | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/guardians-of-the-galaxy-vol-2-2017.pdf) |
| Avengers: Infinity War         | 2018 | ✔️            | 990  | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Avengers:_Infinity_War) |
| Avengers: Endgame              | 2019 | ❌            | 1229 | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/avengers-endgame-2019.pdf) |
