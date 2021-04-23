# marvel-dialogue-nlp
A machine learning project that will use Natural Language Processing (NLP) to classify who says a line of dialogue

<p align="center">
  <img src="https://blog.umhb.edu/wp-content/uploads/2019/06/mcu-1920x1080.jpg" alt="MCU Banner" width="70%" height="70%">
</p>

## Streamlit App
To view an interactive summary of this project, see its [Streamlit app](https://share.streamlit.io/prestondunton/marvel-dialogue-nlp/front_end/front-end.py).

## About the Project
With over 21 different movies, all spanning the same cinematic universe, Marvel movies are an interesting creation of character, dialogue, and plot.  A key defining feature of these movies is the large amount of characters represented and developed.  I wanted to explore NLP, and figured that exploring the dialogue in these 
movies would be a very fun thing to do.  The problem whished to be accomplished in this project is an NLP classification problem.  The goal was to create a model that can predict a character's name given a line of their dialogue from a Marvel Cinematic Universe (MCU) movie.  Data was taken from Marvel released scripts and transformed into labels of names and feature documents of their dialogue.

## Results
In this project, 18 different models were buit and compared.  Models 1-12 use Naive Bayes, SVM, and Random Forest classifiers in different architecture combinations and can be read about in the [old models directory](https://github.com/prestondunton/marvel-dialogue-nlp/tree/master/old%20models) Model 13 is the Naive Bayes classifier with the best performance and presented as the production model.  Models 14-18 are derived from model 13, but manipulated the data or larger architecture to try to achieve better results.  Model 14 is an ensemble method that trains a model for every character and can be read about in the [One vs. Rest Models](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/One%20vs.%20Rest%20Models.ipynb) notebook.  Model 15 allows the use of movie titles and authors as features and can be read about in the [All Features Model](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/All%20Features%20Model.ipynb) notebook. Models 16, 17, and 18 were inspired by the correlation between the number of words in a line and its correct prediction, shown in the section below.  These models attempt to train on less sparse vectors and can be read about in the [Word Count Models](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/Word%20Count%20Models.ipynb) notebook.

Model 13 uses scikit-learn to implement a Naive Bayes Classifier.  Hyperparameter selection is done using cross validation (10 folds).  The model is also evaluated using cross validation (10 folds).  With hyperparameter selection, this results in nested cross validation.  Stop words, which are words that provide no value to predictions (I, you, the, a, an, ...), are not removed from predictions.  Hyperparameter selection showed better performance keeping all words rather than removing NLTK's list of stop words. Words are stemmed using NLTK's SnowballStemmer.  Word counts are also transformed with term frequencies and inverse document frequencies using scikit-learn's implementation.

Model 13 performs with 29.041% balanced accuracy.  The model's performance isn't great, but it's still fun to interact with!  Over the course of the project it's been shown that more data only results in marginal increases in performance.  Above, it's shown that accuracy increases as the number of words in a line increases.  In other words, it seems that spoken dialogue is too short to predict in this case.  The Naive Bayes classifier is a Bag of Words model, meaning that the order of the words is ignored.  By using a Word Embeddings model, which does not ignore the order of words, accuracy could possibly be increased.  Deep learning also might have success on this dataset.

To see the code for the model, see the [Production Model](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/Production%20Model.ipynb) notebook.


## About the Dataset
This repository contains a newly created dataset to train and test models on, as well as several Jupyter Notebooks that describe the process used to create each `.csv`.  This dataset uses a combination of original scripts and transcripts from the Marvel Cinematic Universe (MCU) movies.  The original script `.pdf`s were obtained from [Script Slug](https://www.scriptslug.com/scripts/category/marvel), though other copies of the Marvel released scripts can be found online elsewhere.  Transcripts were taken from [Fandom's Transcripts Wiki](https://transcripts.fandom.com/wiki/Category:Marvel_Transcripts). Transcripts were copied and pasted into `.txt` files, and then processed using `pandas`. See the table below to find out what movies are in the dataset, and where the dialogue came from.

If you spot a mistake in the dataset, please let me know so I can correct it.

If you would like to use the dataset, it is available on [Kaggle](https://www.kaggle.com/pdunton/marvel-cinematic-universe-dialogue).

### MCU.csv
The end file, [mcu.csv](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/mcu.csv), contains columns `character` and `line` that hold the dialogue for several movies from the MCU. There are more columns that provide additional features for context, such as movie titles and author indicators.  See [/data/MCU.ipynb](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/MCU.ipynb) for more details on those features and the dataset's creation. 

### Individual Movies
For individual movies, the `.csv` files found in [/data/cleaned/](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned) should **not** be used.  Instead, load the file [mcu.csv](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/mcu.csv) and use `pandas` or a similar library to select the rows that match a given movie.

### Other Files of Interest
The file [mcu.csv](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/mcu.csv) contains all of the data, but there are other files that might be of interest for anyone using this dataset.  The file [mcu_subset.csv](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/mcu_subset.csv) is a subset of the original data, only containing dialogue for `['TONY STARK', 'STEVE ROGERS', 'NATASHA ROMANOFF', 'THOR', 'NICK FURY', 'PEPPER POTTS', 'BRUCE BANNER', 'JAMES RHODES', 'LOKI', 'PETER PARKER']`, who are the top 10 characters by number of lines, number of words, and movie appearances. 

The file [characters.csv](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/characters.csv) contains statistics summarizing each character's involvement in the MCU.  The movie name columns in that table describe how many lines they have in that movie. 

The file [movies.csv](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/movies.csv) contains metadata about the movies used in this project.  For details on the creation of these files, see [/data/MCU.ipynb](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/MCU.ipynb).

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
| [Thor: Ragnarok](https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/cleaned/ragnarok.csv) | 2017 | ❌ | 961 | [Script Slug](https://www.scriptslug.com/assets/uploads/scripts/thor-ragnorak-2017.pdf) |
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
