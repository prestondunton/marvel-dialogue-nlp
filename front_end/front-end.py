import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer

CHARACTER_IMAGE_PATHS = {"TONY STARK": "./images/tony-stark.png",
                         "ROCKET": "./images/rocket-raccoon.png",
                         "BRUCE BANNER": "./images/bruce-banner.png",
                         "PEPPER POTTS": "./images/pepper-potts.png",
                         "NATASHA": "./images/natasha.png",
                         "LOKI": "./images/loki.png",
                         "PETER QUILL": "./images/peter-quill.png",
                         "STEVE ROGERS": "./images/steve-rogers.png",
                         "THOR": "./images/thor.png"}

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

        self.prediction = None
        self.prediction_conf = None
        self.prediction_probs = None

        self.rank_table = None
        self.hierarchical_rank_table = None

    def render_header(self):
        st.title("Marvel Dialogue Classification")
        st.text("By Preston Dunton")
        st.text("")
        st.text("")

    def render_input_box(self):
        self.input_string = st.text_input('Input String', 'I am Iron Man.')

    def render_prediction(self):
        self.prediction = self.model.predict([self.input_string])
        self.prediction_conf = self.model.predict_proba([self.input_string]).max()
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            st.header("Prediction:")
            st.subheader(self.prediction[0].title())
        with col2:
            st.header("Confidence:")
            st.subheader("{0:.3%}".format(self.prediction_conf))
        with col3:
            st.image(CHARACTER_IMAGE_PATHS[self.prediction[0]], width=200)


    def render_probability_table(self):

        st.header("Probability Table")
        st.text("The table below shows the probabilities the model predicts given a character the input.\n"
                "Each cell holds the probability of predicting its row's character given its column's\n"
                "word. In other words:")
        st.latex("cell(row, column)=p(character|word)")
        st.text("The final column represents the probability our model predicts a character given the\n"
                "entire input string together.  The largest value in this column is our model's\n"
                "prediction.  Some words like 'I' and 'a' are removed because they don't provide any\n"
                "useful information to the model.")
        st.text("By clicking on the names of the columns, you can sort the table and see which\n"
                "character is most likely to say a word.")

        vect = self.model.named_steps['vect']
        tokenizer = vect.build_tokenizer()
        prediction_array = tokenizer(self.input_string)
        prediction_array.append(self.input_string)

        probabilities = pd.DataFrame(self.model.predict_proba(prediction_array).transpose())
        probabilities.columns = prediction_array
        probabilities.columns = [*probabilities.columns[:-1], 'Total Probability']
        probabilities.insert(0, "character", self.model.classes_)
        probabilities.set_index('character', inplace=True)
        probabilities.sort_values(by=['Total Probability'], ascending=False, inplace=True)

        self.prediction_probs = probabilities

        #st.dataframe(self.prediction_probs)
        #st.dataframe(self.prediction_probs.style.apply(self.custom_style, axis=1))
        #st.dataframe(self.prediction_probs.style.apply(self.highlight_max_cell, axis=1))
        st.dataframe(self.prediction_probs.style.background_gradient(cmap=plt.cm.Reds, high=0.35))

    def highlight_max_cell(self, s):
        is_max = s == s.max()
        return ['background-color: #ed1d24' if v else '' for v in is_max]

    def text_color(self, s):
        is_max = s == s.max()
        return ['text-color: white' if v else '' for v in is_max]

    def custom_style(self, row):
        if row.name == self.prediction:
            color = '#ed1d24'
        else:
            color = 'white'

        return ['background-color: %s' % color] * len(row.values)

    def render_rank_table(self):
        ranks = self.prediction_probs.rank(axis=0, method='max', ascending=False)
        ranks.reset_index(inplace=True)

        rank_table = pd.DataFrame()
        for name in ranks.columns:
            column = pd.Series(
                [ranks['character'].to_numpy()[index] for index in ranks[name].sort_values().index],
                name=name)
            rank_table.insert(len(rank_table.columns), name, column, True)

        rank_table.drop(columns=['character'], inplace=True)
        rank_table.index += 1

        self.rank_table = rank_table
        #st.dataframe(rank_table)


    def render_hierarchical_rank_table(self):
        rank_table = pd.DataFrame()

        for word in self.rank_table.columns:
            word_table = pd.DataFrame({'character': self.rank_table[word], 'probability': self.prediction_probs[word].sort_values(ascending=False).to_numpy()})
            rank_table = pd.concat([rank_table, pd.concat({word: word_table}, axis=1)], axis=1)

        self.hierarchical_rank_table = rank_table
        st.dataframe(rank_table)

    def render_model_performance(self):
        st.header("Model Performance")

        y = self.model_predictions['true character']
        yhat = self.model_predictions['predicted character']
        main_characters = self.model_predictions["true character"].value_counts().index.to_numpy()

        st.subheader("Confusion Matrix")

        conf_matrix = pd.DataFrame(metrics.confusion_matrix(y, yhat, labels=main_characters))
        normalized_conf_matrix = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
        normalized_conf_matrix.columns = pd.Series(main_characters, name="Predicted Character")
        normalized_conf_matrix.index = pd.Series(main_characters, name="True Character")

        fig = plt.figure(figsize=(2, 2))
        plt.title("Proportion of a True Character's Examples")
        fig, ax = plt.subplots()
        ax = sns.heatmap(normalized_conf_matrix, annot=True, fmt='.2f', cmap=plt.cm.Reds)

        st.pyplot(fig)

        st.subheader("Accuracy by Character (Recall)")
        recalls = pd.DataFrame(np.diagonal(normalized_conf_matrix.to_numpy()), index=main_characters, columns=["accuracy"])
        recalls.sort_values(by="accuracy", ascending=False, inplace=True)
        recalls.loc['mean'] = recalls.mean()
        st.dataframe(recalls)

        st.subheader("Model's Balanced Accuracy: {0:.3%}".format(metrics.balanced_accuracy_score(y, yhat)))

    def render_model_predictions(self):
        st.header("Model Predictions")

        table = self.model_predictions
        table['correct prediction'] = table['true character'] == table['predicted character']
        table['correct prediction'] = table['correct prediction'].replace({0: 'No', 1: "Yes"})

        table.sort_values(by=["correct prediction"], inplace=True, ascending=False)
        table.reset_index(drop=True)

        st.table(table)

    def render_app(self):
        st.set_page_config(page_title='Marvel Dialogue Classification', layout='centered', \
                           initial_sidebar_state='auto', page_icon="./images/marvel-favicon.png")

        self.render_header()
        self.render_input_box()
        self.render_prediction()
        self.render_probability_table()

        #st.header("Rank Table")
        #self.render_rank_table()
        #self.render_hierarchical_rank_table()

        self.render_model_performance()
        self.render_model_predictions()

app = Application()
app.render_app()