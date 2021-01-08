import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

CHARACTER_IMAGE_PATHS = {"TONY STARK": "./images/tony-stark.png",
                         "BRUCE BANNER": "./images/bruce-banner.png",
                         "PEPPER POTTS": "./images/pepper-potts.png",
                         "NATASHA ROMANOFF": "./images/natasha.png",
                         "LOKI": "./images/loki.png",
                         "STEVE ROGERS": "./images/steve-rogers.png",
                         "THOR": "./images/thor.png",
                         "NICK FURY": "./images/nick-fury.png",
                         "PETER PARKER": "./images/peter-parker.png",
                         "JAMES RHODES": "./images/james-rhodes.png"}

class StemCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemCountVectorizer, self).build_analyzer()

        return lambda document: (
        [SnowballStemmer('english', ignore_stopwords=True).stem(word) for word in analyzer(document)])

class Application():
    def __init__(self):
        self.input_string = None
        self.model = load('./production_model.joblib')
        self.model_predictions = load('./production_predictions.joblib')
        self.character_correlations = load('./character_correlations.joblib')
        self.main_characters = self.model_predictions["true character"].value_counts().index.to_numpy()

        self.prediction = None
        self.prediction_conf = None
        self.prediction_probs = None

        self.rank_table = None
        self.hierarchical_rank_table = None
        
        self.recalls = pd.read_csv("./production_recalls.csv")
        self.confusion_matrix = None

    def render_header(self):
        
        st.markdown('<style>.text{font-family: "IBM Plex Mono", monospace; white-space: pre;font-size: 0.8rem; overflow-x: auto;}</style>', unsafe_allow_html=True)
        
        st.title("Marvel Dialogue Classification")
        st.text("By Preston Dunton")
        st.text("")
        st.text("")
        
        st.text("This page presents a machine learning project on dialogue from Marvel's movies.  Below,\n"
                "you'll find an interactive interface for predictions, details about the model, metrics of\n"
                "the model's performance, insights into the Marvel Cinematic Universe (MCU), and\n"
                "interactive samples of the model's predictions.")
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
            st.image(CHARACTER_IMAGE_PATHS[self.prediction[0]], width=200)
            
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
        st.markdown('<p class="text">This project uses <a href="https://scikit-learn.org/stable/" target="_blank">scikit-learn</a> to implement a <a href="https://www.youtube.com/watch?v=O2L2Uv9pdDA" target="_blank">Naive Bayes Classifier</a>.  Hyperparameter\n'
                    'selection is done using cross validation (5 folds).  The model is also evaluated using\n'
                    'cross validation (5 folds).  With hyperparameter selection, this results in nested cross\n'
                    'validation.  Stop words, which are words that provide no value to predictions (I, you,\n'
                    'the, a, an, ...), are not removed from predictions.  Hyperparameter selection showed\n'
                    'better performance keeping all words rather than removing <a href="https://www.geeksforgeeks.org/removing-stop-words-nltk-python/" target="_blank">NLTK\'s list of stop words</a>.\n'
                    'Words are stemmed using <a href="https://www.nltk.org/_modules/nltk/stem/snowball.html" target="_blank">NLTK\'s SnowballStemmer</a>.  Word counts are also transformed into\n'
                    'term frequencies using scikit-learn\'s implementation.</p>', unsafe_allow_html=True)
        st.markdown('<p class="text">To see the code for the model, see <a href="https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/Production%20Models.ipynb" target="_blank">this Jupyter Notebook</a></p>', unsafe_allow_html=True)
        
        st.subheader("Dataset Details")
        st.markdown('<p class="text">The dataset used was created for this project by parsing the Marvel released script /\n'
                    'online transcript for 18 movies.  See the <a href="https://github.com/prestondunton/marvel-dialogue-nlp" target="_blank">repository\'s README.md</a> for a table of which\n'
                    'movies are included and which are not, as well as more details.</p>', unsafe_allow_html=True)
        
        st.subheader("Why these characters?")
        st.markdown("<p class='text'>While the dataset contains the dialogue for all 652 character, most of which are just\n"
                "movie extras, trying to predict a large number of characters results in such poor\n"
                "performance that the model isn't useful or fun in any way.  The ten characters\n"
                "used are the top ten characters by number of lines in the dataset, number of words\n"
                "in the dataset, and number of movie appearances.  See <a href='https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/data/MCU.ipynb' target='_blank'>this Jupyter Notebook</a> for details\n"
                    "on this dataset's creation and the calculations used to select these characters.</p>", unsafe_allow_html=True)

    def render_model_performance(self):
        st.header("Model Performance")

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
        st.markdown('<p class="text">To see the code for these metrics, and more metrics, see <a href="https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/Production%20Models.ipynb" target="_blank">this Jupyter Notebook</a></p>', unsafe_allow_html=True)
        
    def render_confusion_matrix(self):
        y = self.model_predictions['true character']
        yhat = self.model_predictions['predicted character']

        st.subheader("Confusion Matrix")

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
        
        st.subheader("Accuracy Vs. Words")
        
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

    def render_model_insights(self):
        st.header("MCU Insights")
        
        st.subheader("Character Correlation")
        
        fig = plt.figure(figsize=(2, 2))
        fig, ax = plt.subplots()
        ax = sns.heatmap(self.character_correlations, annot=True, fmt='.2f', cmap=plt.cm.Reds)

        st.pyplot(fig)
        
    
    def render_model_predictions(self):
        st.header("Model Predictions")
        
        st.text("Use the multiselects to view how the model performed on lines from the movies.  These\n"
                "predictions were created using cross-validation (5 fold), so no example is predicted\n"
                "with a model that saw the example in training.")

        table = self.model_predictions
        table['correct prediction'] = table['true character'] == table['predicted character']
        table['correct prediction'] = table['correct prediction'].replace({0: 'No', 1: "Yes"})        

        true_character_filter = st.multiselect("True Character", list(self.main_characters), ["PETER PARKER"])
        pred_character_filter = st.multiselect("Predicted Character", list(self.main_characters), ["PETER PARKER"])
        movie_filter = st.multiselect("Movie", list(self.model_predictions['movie'].unique()), ['Spider-Man: Homecoming', "Captain America: Civil War"])
        
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
                           initial_sidebar_state='auto', page_icon="./images/marvel-favicon.png")

        self.render_header()
        
        st.image("./images/horizontal_line.png", use_column_width=True)
        self.render_interactive_prediction()

        st.text(" ")
        st.text(" ")
        st.text(" ")
        
        st.image("./images/horizontal_line.png", use_column_width=True)
        self.render_about_the_model()
               
        st.text(" ")
        st.text(" ")
        st.text(" ")

        st.image("./images/horizontal_line.png", use_column_width=True)
        self.render_model_performance()

        st.text(" ")
        st.text(" ")
        st.text(" ")
        
        st.image("./images/horizontal_line.png", use_column_width=True)
        self.render_model_insights()
        
        st.text(" ")
        st.text(" ")
        st.text(" ")

        st.image("./images/horizontal_line.png", use_column_width=True)
        self.render_model_predictions()

app = Application()
app.render_app()