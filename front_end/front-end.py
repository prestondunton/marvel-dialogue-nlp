import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import SnowballStemmer
nltk.download('stopwords')

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

CHARACTER_IMAGE_PATHS = {"TONY STARK": "/images/tony-stark.png",
                         "BRUCE BANNER": "/images/bruce-banner.png",
                         "PEPPER POTTS": "/images/pepper-potts.png",
                         "NATASHA ROMANOFF": "/images/natasha.png",
                         "LOKI": "/images/loki.png",
                         "STEVE ROGERS": "/images/steve-rogers.png",
                         "THOR": "/images/thor.png",
                         "NICK FURY": "/images/nick-fury.png",
                         "PETER PARKER": "/images/peter-parker.png",
                         "JAMES RHODES": "/images/james-rhodes.png"}

class StemCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemCountVectorizer, self).build_analyzer()

        return lambda document: (
        [SnowballStemmer('english', ignore_stopwords=True).stem(word) for word in analyzer(document)])

class Application():
    def __init__(self):
        self.file_path = "/app/marvel-dialogue-nlp/front_end"
        
        self.input_string = None
        self.model = load(self.file_path + '/production_model.joblib')
        self.model_predictions = load(self.file_path + '/production_predictions.joblib')
        self.character_similarity = load(self.file_path + '/character_similarity.joblib')
        self.main_characters = self.model_predictions["true character"].value_counts().index.to_numpy()

        self.prediction = None
        self.prediction_conf = None
        self.prediction_probs = None

        self.rank_table = None
        self.hierarchical_rank_table = None
        
        self.recalls = pd.read_csv(self.file_path + "/production_recalls.csv")
        self.confusion_matrix = None

    def render_header(self):
        
        st.markdown('<style>.text{font-family: "IBM Plex Mono", monospace; white-space: pre;font-size: 0.8rem; overflow-x: auto;}</style>', unsafe_allow_html=True)
        
        st.title("Marvel Dialogue Classification")
        st.text("By Preston Dunton")
        st.text("")
        st.text("")
        
        st.text("This page presents a machine learning project on dialogue from Marvel's movies.  Below,\n"
                "you'll find an interactive interface for predictions, details about the model and other\n"
                "models, metrics of the model's performance, insights into the Marvel Cinematic Universe\n"
                "(MCU), and interactive samples of the model's predictions.")
        st.markdown('<p class="text">For more about the project see its <a class="text" href="https://github.com/prestondunton/marvel-dialogue-nlp" target="_blank">GitHub repository</a>.  Feel free to contact me at\n<a class="text" href="mailto:preston.dunton@gmail.com">preston.dunton@gmail.com</a>.</p>', unsafe_allow_html=True)
        st.markdown('', unsafe_allow_html=True)
        
        st.text("")
        st.text("")

    def render_interactive_prediction(self):
        
        st.header("Interactive Prediction")
        st.text("Type in a line to see which character is predicted to say it!")
        
        self.input_string = st.text_input('Input Line', 'I am Iron Man.')
        
        self.prediction = self.model.predict([self.input_string])
        self.prediction_conf = self.model.predict_proba([self.input_string]).max()
        col1, col2, col3 = st.beta_columns(3)
        
        st.markdown('<style>.prediction{color: red; font-size: 24px; font-weight: bold}</style>', unsafe_allow_html=True)
        
        with col1:
            st.subheader("Prediction:")
            st.markdown('<p class="prediction">' + self.prediction[0].title() + '</p>', unsafe_allow_html=True)
        with col2:
            st.subheader("Confidence:")
            st.markdown('<p class="prediction">' + "{0:.3%}".format(self.prediction_conf) + '</p>', unsafe_allow_html=True)
        with col3:
            st.image(self.file_path + CHARACTER_IMAGE_PATHS[self.prediction[0]], width=200)
            
        self.render_probability_table()


    def render_probability_table(self):

        #st.subheader("Probability Table")

        vect = self.model.named_steps['vect']
        tokenizer = vect.build_tokenizer()
        prediction_array = tokenizer(self.input_string)
        prediction_array.append(self.input_string)

        probabilities = pd.DataFrame(self.model.predict_proba(prediction_array).transpose())
        probabilities.columns = prediction_array
        probabilities.columns = [*probabilities.columns[:-1], 'Combined Probability']
        probabilities.insert(0, "character", self.model.classes_)
        probabilities.set_index('character', inplace=True)
        probabilities.sort_values(by=['Combined Probability'], ascending=False, inplace=True)
        
        used_column_names = []
        column_names = probabilities.columns.to_numpy()
        for i in range(0,len(column_names)):
            while column_names[i] in used_column_names:
                column_names[i] = column_names[i] + " "
                
            used_column_names.append(column_names[i])
            
        probabilities.columns = column_names

        self.prediction_probs = probabilities
        
        st.dataframe(self.prediction_probs.style.background_gradient(cmap=plt.cm.Reds, high=0.35))
        
        st.text("The table above shows the probabilities the model predicts given a character the input.\n"
                "Each cell holds the probability of predicting its row's character given its column's\n"
                "word. In other words:")
        st.latex("cell(row, column)=P(character|word)")
        st.text("The final column represents the probability our model predicts a character given the\n"
                "entire input string together.  The character with the largest value in this column\n"
                "is our model's prediction.  One character words like 'I' and 'a' are removed because\n"
                "they don't provide any useful information to the model.  No other words are removed.")
        st.text("By clicking on the names of the columns, you can sort the table and see which\n"
                "character is most likely to say a word.")

    def render_about_the_model(self):
        st.header("About the Model")
        
        
        st.subheader("Implementation Details")
        st.markdown('<p class="text">This project uses <a href="https://scikit-learn.org/stable/" target="_blank">scikit-learn</a> to implement a <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier" target="_blank">Naive Bayes Classifier</a>.  Hyperparameter\n'
                    'selection is done using cross validation (10 folds).  The model is also evaluated using\n'
                    'cross validation (10 folds).  With hyperparameter selection, this results in nested cross\n'
                    'validation.  Stop words, which are words that provide no value to predictions (I, you,\n'
                    'the, a, an, ...), are not removed from predictions.  Hyperparameter selection showed\n'
                    'better performance keeping all words rather than removing <a href="https://www.geeksforgeeks.org/removing-stop-words-nltk-python/" target="_blank">NLTK\'s list of stop words</a>.\n'
                    'Words are stemmed using <a href="https://www.nltk.org/_modules/nltk/stem/snowball.html" target="_blank">NLTK\'s SnowballStemmer</a>.  Word counts are also transformed with\n'
                    'term frequencies and inverse document frequencies using scikit-learn\'s implementation.</p>', unsafe_allow_html=True)
        st.markdown('<p class="text">To see the code for the model, see the <a href="https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/Production%20Model.ipynb" target="_blank">Production Model</a> Jupyter Notebook.</p>', unsafe_allow_html=True)
        
        st.subheader("Dataset Details")
        st.markdown('<p class="text">The dataset used was created for this project by parsing the Marvel released script /\n'
                    'online transcript for 18 movies.  See the <a href="https://github.com/prestondunton/marvel-dialogue-nlp" target="_blank">repository\'s README.md</a> for a table of which\n'
                    'movies are included and which are not, as well as more details.  If you spot an error\n'
                    'in the data, please contact me so I can fix it.</p>', unsafe_allow_html=True)
        st.markdown('<p class="text">To see an in depth analysis of the dataset, see the <a href="https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/Dataset%20Analysis.ipynb" target="_blank">Dataset Analysis</a> Jupyter Notebook.</p>', unsafe_allow_html=True)
        st.markdown("<p class='text'>If you would like to use the dataset, it is available on <a href='https://www.kaggle.com/pdunton/marvel-cinematic-universe-dialogue' target=_blank'>Kaggle</a>.", unsafe_allow_html=True)
        
        st.subheader("Why these characters?")
        st.markdown("<p class='text'>While the dataset contains the dialogue for all 652 character, most of which are just\n"
                "movie extras, trying to predict a large number of characters results in such poor\n"
                "performance that the model isn't useful or fun in any way.  The ten characters\n"
                "used are the top ten characters by number of lines in the dataset, number of words\n"
                "in the dataset, and number of movie appearances.</p>", unsafe_allow_html=True)
        st.markdown('<p class="text">For details on this dataset\'s creation and the calculations used to select these\n'
                    'characters, see the <a href="https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/MCU.ipynb" target="_blank">MCU</a> Jupyter Notebook.</p>', unsafe_allow_html=True)
        
        st.subheader("Other Models")
        st.markdown("<p class='text'>In this project, 18 different models were buit and compared.  Models 1-12 use Naive Bayes,\n"
                    "SVM, and Random Forest classifiers in different architecture combinations and can be read\n"
                    "about in the <a href='https://github.com/prestondunton/marvel-dialogue-nlp/tree/master/old%20models' target='_blank'>old models directory</a>. Model 13 is the Naive Bayes classifier with the best\n"
                    "performance and presented here as the production model.  Models 14-18 are derived from\n"
                    "model 13, but manipulated the data or larger architecture to try to achieve better\n"
                    "results.  Model 14 is an ensemble method that trains a model for every character and can\n"
                    "be read about in the <a href='https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/One%20vs.%20Rest%20Models.ipynb' target='_blank'>One vs. Rest Models</a> notebook.  Model 15 allows the use of movie\n"
                    "titles and authors as features and can be read about in the <a href='https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/All%20Features%20Model.ipynb' target='_blank'>All Features Model</a> notebook.\n"
                    "Models 16, 17, and 18 were inspired by the correlation between the number of words in a\n"
                    "line and its correct prediction, shown in the section below.  These models attempt to\n"
                    "train on less sparse vectors and can be read about in the <a href='https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/Word%20Count%20Models.ipynb' target='_blank'>Word Count Models</a> notebook.</p>", unsafe_allow_html=True)

    def render_model_performance(self):
        st.header("Model Performance")
        
        st.markdown('<p class="text">To see the code for these metrics, and more metrics, see the <a href="https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/Production%20Model.ipynb" target="_blank">Production Model</a> Jupyter \n'
                    'Notebook.</p>', unsafe_allow_html=True)

        self.render_confusion_matrix()
        self.render_recalls()
        self.render_accuracy_by_words()

        y = self.model_predictions['true character']
        yhat = self.model_predictions['predicted character']
        st.subheader("Model's Balanced Accuracy: {0:.3%}".format(metrics.balanced_accuracy_score(y, yhat)))
        
        st.text("The model's performance isn't great, but it's still fun to interact with!  Over the course\n"
                "of the project it's been shown that more data only results in marginal increases in\n"
                "performance.  Above, it's shown that accuracy increases as the number of words in a line\n"
                "increases.  In other words, it seems that spoken dialogue is too short to predict in\n"
                "this case.  The Naive Bayes classifier is a Bag of Words model, meaning that the order\n"
                "of the words is ignored.  By using a Word Embeddings model, which does not ignore the\n"
                "order of words, accuracy could possibly be increased.  Deep learning also might have\n"
                "success on this dataset.")
        
    def render_confusion_matrix(self):
        y = self.model_predictions['true character']
        yhat = self.model_predictions['predicted character']

        st.subheader("Confusion Matrix")
        
        st.text("The plot below summarizes the predictions of our model.  Each cell represents the\n"
                "proportion of all of a true character's examples that are predicted as a character.\n"
                "In other words, each row adds up to 1.0, and the cells can be seen as what percent of\n"
                "examples are predicted as a given character.")
        st.text("The diagonal elements represent examples that our model correctly predicts, as well as\n"
                "the recall for that character.")

        conf_matrix = pd.DataFrame(metrics.confusion_matrix(y, yhat, labels=self.main_characters))
        normalized_conf_matrix = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
        normalized_conf_matrix.columns = pd.Series(self.main_characters, name="Predicted Character")
        normalized_conf_matrix.index = pd.Series(self.main_characters, name="True Character")
        
        self.confusion_matrix = normalized_conf_matrix

        fig = plt.figure(figsize=(2, 2))
        fig, ax = plt.subplots()
        ax = sns.heatmap(normalized_conf_matrix, annot=True, fmt='.2f', cmap=plt.cm.Reds)

        st.pyplot(fig)
        
    def render_recalls(self):
        st.subheader("Accuracy by Character (Recall)")
        st.text("Given a line we'd like to predict from a given character, here's how often we can expect\n"
                "our model to be correct.")
        
        self.recalls.set_index("Unnamed: 0", drop=True, inplace=True)
        st.dataframe(self.recalls)
        
        
    def render_accuracy_by_words(self):
        
        st.subheader("Performance Vs. Words")
        
        st.text("Do examples with more words (longer examples) get classified correctly more often?\n"
                "Let's do a linear regression test.")
        
        def abline(intercept, slope, col):
            """Plot a line from slope and intercept"""
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, '-', color=col)
       
        regression_data = self.model_predictions.copy(deep=True)
        
        regression_data['words'] = regression_data['line'].str.split(" ").str.len()
        regression_data['correct_prediction'] = (regression_data['true character'] == regression_data['predicted character']).astype('int64')
        
        reg_model = smf.ols('regression_data["correct_prediction"] ~ regression_data["words"]', data=regression_data).fit()
        
        fig = plt.figure(figsize=(2, 2))
        fig, ax = plt.subplots()

        ax = plt.scatter(x = regression_data['words'].to_numpy(),
                y = regression_data['correct_prediction'].to_numpy(),
                color='black')

        abline(reg_model.params[0], reg_model.params[1], 'red')

        conf_pred_intervals = reg_model.get_prediction(regression_data['words']).summary_frame()

        plt.fill_between(regression_data['words'].to_numpy(), conf_pred_intervals['mean_ci_lower'], conf_pred_intervals['mean_ci_upper'], alpha=0.3, color='red')

        plt.grid()
        plt.ylim(-0.1,1.2)
        plt.title('Performance vs. Length of Words (with 95% CI Band)')
        plt.xlabel('words')
        plt.ylabel('accuracy')
        
        st.pyplot(fig)
        
        st.markdown("<p class=\"text\">Using a t test and a confidence level of 95% (α=0.05), we <text class=\"text\" style=\"font-weight: bold\">reject</text> the null hypothesis that\n"
                    "there is no relationship between the number of words in an example and our model's\n"
                    "performance on it (t=10.382, p&lt0.001).  The estimated average difference in the chance\n"
                    "of correct prediction for a one word change is 0.46%.</p>", unsafe_allow_html=True)

        st.markdown("<p class='text'>This correlation does not necessarily mean causation, and in fact was investigated to be\n"
                    "just correlation.  A longer monologue / soliloquy from a character might included key\n"
                    "vocabulary words that helps the model identify the character better.  The <a href='https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/Word%20Count%20Models.ipynb' target='_blank'>Word Count\n"
                    "Models</a> notebook manipulates the data in three different ways in order to try and train\n"
                    "models on vectors with more words (less sparse).  These models did not perform better than\n"
                    "the production model, meaning that this correlation is most likely not causation.</p>", unsafe_allow_html=True)

    def render_mcu_insights(self):
        st.header("MCU Insights")
        
        st.markdown('<p class="text">For the code used to infer these insights, see the <a href="https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/MCU%20Insights.ipynb" target="_blank">MCU Insights</a> Jupyter Notebook.</p>', unsafe_allow_html=True)
        
        self.render_character_similarity()
        self.render_character_development()
        
    def render_character_similarity(self):
        st.subheader("Character Similarity and Peter Parker")
        st.text("The similarity score used here is the dot product between the unit word count vectors of\n"
                "every word a character has ever said.  To get this score, we just count how many times a\n"
                "character has said every word, divide the vector by it's norm, and then do a dot product\n"
                "with another character's vector.")
        st.text("It should be noted that the confusion matrix above could also be interpreted as a measure\n"
                "of similarity.  Characters who are easily confused with each other could be seen as\n"
                "similar.  For example, Thor and Loki are easily confused with each other by the model,\n"
                "which makes sense because they are both Asgardian.")
        
        fig = plt.figure(figsize=(2, 2))
        fig, ax = plt.subplots()
        ax = sns.heatmap(self.character_similarity, annot=True, fmt='.2f', cmap=plt.cm.Reds)

        st.pyplot(fig)
        
        st.markdown('<p class="text" style="font-weight: bold;">Why are these similarity scores so high?</p>', unsafe_allow_html=True)
        st.markdown('<p class="text">The characters are most likely so similar with each other because they are all speaking\n'
                    'English, which like any language, is very structured and systematic.  According to <a href="https://www.youtube.com/watch?v=fCn8zs912OE" target="_blank">Michael\n'
                    'Steven\'s discussion</a> of Zipf\'s Law and the Pareto Distribution, the top 20% of words (words\n'
                    'like "the" "I" "he" "she") make up approximately 80% of all speech.</p>', unsafe_allow_html=True)  

        st.markdown('<p class="text" style="font-weight: bold;">Why is Peter Parker so unique compared to the other characters?</p>', unsafe_allow_html=True)
        
        st.text("Peter Parker only has 4 movie appearances, which is less than all the other characters\n"
                "(9, 10, 7, 10, 7, 7, 7, 6, 8).  However, this doesn't seem to have an effect on the mean,\n"
                "sum, and standard deviation of his word count vector. These statistics do not seem to be\n"
                "out of the ordinary compared to the other characters.  Furthermore, Peter Parker does not\n"
                "seem to have an unusual vocabulary size or number of words unique to himself.  ")

        st.text("Another hypothesis on why Peter Parker is so unique is his age.  He is much younger than\n"
                "all the other characters used, and therefore might talk about different topics, like high\n"
                "school (homecoming, field trips, homework, ...).")

        st.text("Finally, Peter Parker might be so unique because he's not part of the larger Avengers\n"
                "team.  The \"friendly neighborhood Spider-Man\" deals with much smaller problems than\n"
                "the other Avengers.  For example, the Avengers team might be more likely to deal with\n"
                "governments, aliens, SHIELD, and large scale enemies than Peter.  One counter argument to\n"
                "this hypothesis is Pepper Potts, who is also not part of the Avengers team and also has\n"
                "high similarity with the other characters.  However, her close interaction with Tony\n"
                "Stark might make her discuss these same issues when she speaks.  More evidence for this\n"
                "can be seen in the confusion matrix / recalls.  Peter Parker performs the highest out of\n"
                "any of the characters, which suggests that his dialogue style is the most identifiable\n"
                "or unique.")

        st.markdown("<p class=\"text\">The probable answer is that Peter Parker is so unique for all of the reasons mentioned\n"
                "above together. Peter Parker is arguably the most unique character in the MCU, which is\n"
                "why he is such a fan favorite.  If a trascript for <i>Spider-Man: Far Frome Home</i> was\n"
                    "completed and added to the dataset, it would be interesting to see how that affects this\n"
                    "similarity.</p>", unsafe_allow_html=True)
        
    def render_character_development(self):
        st.subheader("Character Development and Thor")
        
        st.markdown("<p class='text'>An interesting extension of this project would be to deeply explore the question: <p class='text' style='font-weight: bold'>Are there quantifiable differences between a character in different movies or written\n"
                    "under different authors?</p></p>", unsafe_allow_html=True)
        st.markdown("<p class='text'>Though I have little experience in unsupervised learning, I wonder if clustering would\n"
                    "reveal any insights.  For example, if we did clustering with more than 10 clusters, would\n"
                    "the clusters be the characters in different movies?  The approach below for Thor uses\n"
                    "supervised learning and exploratory data analysis.  The code can be found in the Jupyter\n"
                    "Notebook for this section.</p>", unsafe_allow_html=True)
        
        st.markdown("<p class='text' style='font-weight: bold'>Thor's Character Development</p>", unsafe_allow_html=True)
        st.markdown("<p class='text'>When creating this section, I wanted to explore Thor's character development specifically.\n"
                    "Thor in <i>Thor, Thor: The Dark World, The Avengers</i>, and <i>Avengers: Age of Ultron</i>, is very\n"
                    "different from Thor in <i>Thor: Ragnarok, Avengers: Infinity War</i>, and <i>Avengers: Endgame.\n"
                    "Thor: Ragnorak</i> is a turning point for Thor, as he takes on a more comedic role than\n"
                    "previously.  Chris Hemsworth, the producers, and the screen writers wanted to take the\n"
                    "character in a different direction in this movie, and it definitely shows in the following\n"
                    "movies as well.  In other words, his character in the first half of all his movies are\n"
                    "more serious than his character in the second half of all his movies.</p>", unsafe_allow_html=True)
        st.markdown("<p class='text'>By retraining the presented model on Thor's dialogue and making the labels movies\n"
                    "\"PRE RAGNAROK\" and \"POST RAGNAROK\" (post includes <i>Thor: Ragnarok</i>), I was able to obtain\n"
                    "interpretable metrics.  The model used can determine if a line came from a movie pre/post\n"
                    "<i>Thor: Ragnarok</i> with 70.705% balanced accuracy. This is the highest score found when using\n"
                    "the same technique on other characters.  I would say that this accuracy is indicative of\n"
                    "the changes to Thor we observe, but I wouldn't use this score to make larger inferences\n"
                    "about character development.</p>", unsafe_allow_html=True)
        st.text("See the notebook for this section for more about these metrics, and to see them applied\n"
                "to other characters.")
        
    def render_model_predictions(self):
        st.header("Model Predictions")
        
        st.text("Use the multiselects to view how the model performed on lines from the movies.  These\n"
                "predictions were created using cross-validation (10 fold), so no example is predicted\n"
                "with a model that saw the example in training.")

        table = self.model_predictions
        table['correct prediction'] = table['true character'] == table['predicted character']
        table['correct prediction'] = table['correct prediction'].replace({0: 'No', 1: "Yes"})        

        true_character_filter = st.multiselect("True Character", list(self.main_characters), ["PETER PARKER"])
        pred_character_filter = st.multiselect("Predicted Character", list(self.main_characters), ["PETER PARKER"])
        movie_filter = st.multiselect("Movie", list(self.model_predictions['movie'].unique()), ["Captain America: Civil War", 'Avengers: Endgame'])
        
        if len(true_character_filter) == 0:
            true_character_filter = self.main_characters
        if len(pred_character_filter) == 0:
            pred_character_filter = self.main_characters
        if len(movie_filter) == 0:
            movie_filter = self.model_predictions['movie'].unique()
        
        st.table(table[table['true character'].isin(true_character_filter) &
                       table['predicted character'].isin(pred_character_filter) &
                       table['movie'].isin(movie_filter)])
        
    def render_app(self):
        st.set_page_config(page_title='Marvel Dialogue Classification', layout='centered', \
                           initial_sidebar_state='auto', page_icon=self.file_path + "/images/marvel-favicon.png")

        self.render_header()
        
        st.image(self.file_path + "/images/horizontal_line.png", use_column_width=True)
        self.render_interactive_prediction()

        st.text(" ")
        st.text(" ")
        st.text(" ")
        
        st.image(self.file_path + "/images/horizontal_line.png", use_column_width=True)
        self.render_about_the_model()
               
        st.text(" ")
        st.text(" ")
        st.text(" ")

        st.image(self.file_path + "/images/horizontal_line.png", use_column_width=True)
        self.render_model_performance()

        st.text(" ")
        st.text(" ")
        st.text(" ")
        
        st.image(self.file_path + "/images/horizontal_line.png", use_column_width=True)
        self.render_mcu_insights()
        
        st.text(" ")
        st.text(" ")
        st.text(" ")

        st.image(self.file_path + "/images/horizontal_line.png", use_column_width=True)
        self.render_model_predictions()

app = Application()
app.render_app()