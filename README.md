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
This repository contains a newly created dataset to train and test models on, as well as several Jupyter Notebooks that describe the process used to create each `.csv`.  These Jupyter notebooks explain the process of parsing the `.pdf`s with the `pandas` library.  The end file, [mcu.csv](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/mcu.csv), contains columns `character` and `line` that hold the dialogue for several movies from the MCU. There are more columns that provide additional features for context, but were not used in this project.  See [/data/MCU.ipynb](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/MCU.ipynb) for more details on those features. For individual movies, the corresponding `.csv` can be found in [/data/cleaned/](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned) and contain columns `character` and `line`.  Each movie file was created using the same partially automated process, though improvements were found as more movies were processed.  To see the code that generated each `.csv`, look for the Jupyter Notebook with the same movie name. 

This dataset uses a combination of original scripts and transcripts from the MCU movies.  The original script `.pdf`s were obtained from [Script Slug](https://www.scriptslug.com/scripts/category/marvel), though other copies of the Marvel released scripts can be found online elsewhere.  Transcripts were taken from [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Category:Marvel_Transcripts). Transcripts were copied and pasted into `.txt` files, and then processed using `pandas`. Creating each `.csv` took quite a bit of time, so currently, this dataset only contains the movies below.  The table below contains metadata about the dataset and each movie's source.  This table is stored in [movies.csv](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/movies.csv).

If you spot a mistake in the dataset, please let me know so I can correct it.

### Movies Included

| Movie                               | Year | Is Transcript | Lines | Source Link | CSV Issues |
| ----------------------------------- | ---- | ------------- | ----- | ----------- | ------ |
| [Iron Man](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/iron_man.csv) | 2008 | ❌ | 834 | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/iron-man-2008.pdf) |
| [Iron Man 2](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/iron_man_2.csv) | 2010 | ✔️ | 1010 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Iron_Man_2) | Not proofread |
| [Thor](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/thor.csv) | 2011 | ❌ | 1007 | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/thor-2011.pdf) |
| [Captain America: The First Avenger](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/captain_america.csv) | 2011 | ✔️ | 688 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Captain_America:_The_First_Avenger) | Not proofread |
| [The Avengers](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/avengers.csv) | 2012 | ❌ | 1027 | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/the-avengers-2012.pdf) |
| [Iron Man 3](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/iron_man_3.csv) | 2013 | ✔️ | 1043 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Iron_Man_3) | Not proofread |
| [Thor: The Dark World](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/thor_dark_world.csv) | 2013 | ✔️ | 734 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Thor:_The_Dark_World) | Not proofread, transcript duplication |
| [Captain America: The Winter Soldier](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/winter_soldier.csv) | 2014 | ✔️ | 841 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Captain_America:_The_Winter_Soldier) | Not proofread |
| [Avengers: Age of Ultron](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/age_of_ultron.csv) | 2015 | ✔️ | 980 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Avengers:_Age_of_Ultron) | Not proofread |
| [Ant Man](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/ant_man.csv) | 2015 | ✔️ | 867 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Ant-Man) | Not proofread |
| [Captain America: Civil War](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/civil_war.csv) | 2016 | ✔️ | 987 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Captain_America:_Civil_War) | Not proofread |
| [Guardians of the Galaxy: Vol 2](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/guardians_2.csv) | 2017 | ❌ | 993 | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/guardians-of-the-galaxy-vol-2-2017.pdf) | |
| [Spider-Man: Homecoming](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/spider_man_homecoming.csv) | 2017 | ✔️ | 1558 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Spider-Man:_Homecoming) | Not proofread |
| [Thor: Ragnarok](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/ragnorak.csv) | 2017 | ❌ | 961 | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/thor-ragnorak-2017.pdf) |
| [Black Panther](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/black_panther.csv) | 2018 | ❌ | 834 | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/black-panther-2018.pdf) |
| [Avengers: Infinity War](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/infinity_war.csv) | 2018 | ✔️ | 990 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Avengers:_Infinity_War) |
| [Captain Marvel](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/captain_marvel.csv) | 2019 | ✔️ | 775 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Captain_Marvel_(2019)) | Not proofread |
| [Avengers: Endgame](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/avengers_endgame.csv) | 2019 | ❌ | 1229 | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/avengers-endgame-2019.pdf) |

### Movies not included
| Movie                               | Year | Source Link | Transcript Issues |
| ------------------------- | ---- | ----------- | ------ |
| The Incredible Hulk       | 2008 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/The_Incredible_Hulk) | Poor / messy transcript |
| Guardians of the Galaxy   | 2014 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Guardians_of_the_Galaxy) | Poor / messy transcript |
| Doctor Strange            | 2016 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Doctor_Strange) | Poor / messy transcript |
| Ant-Man and the Wasp      | 2018 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Ant-Man_and_the_Wasp) | Incomplete transcript |
| Spider-Man: Far From Home | 2019 | [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Spider-Man:_Far_From_Home) | Incomplete transcript |

## Libaries Used
- numpy
- pandas
- sklearn
- nltk
- matplotlib
- seaborn
- statsmodels
