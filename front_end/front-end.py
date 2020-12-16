import streamlit as st
import pandas as pd
from joblib import load

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
        self.prediction = None
        self.prediction_conf = None

    def render_header(self):
        st.title("Marvel Dialogue Classification")
        st.text("By Preston Dunton")
        st.text("")
        st.text("")

    def render_input_box(self):
        self.input_string = st.text_input('Input String', 'I am Iron Man')

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
        vect = StemCountVectorizer(binary=False)
        tokenizer = vect.build_tokenizer()
        prediction_array = tokenizer(self.input_string)
        prediction_array.append(self.input_string)

        probabilities = pd.DataFrame(self.model.predict_proba(prediction_array).transpose())
        probabilities.columns = prediction_array
        probabilities.columns = [*probabilities.columns[:-1], 'probability']
        probabilities.insert(0, "character", self.model.classes_)

        st.dataframe(probabilities.style.apply(self.custom_style, axis=1))

    def custom_style(self, row):
        if row.values[0] == self.prediction:
            color = 'red'
        else:
            color = 'white'

        return ['background-color: %s' % color] * len(row.values)


    def render_app(self):
        self.render_header()
        self.render_input_box()
        self.render_prediction()
        self.render_probability_table()


app = Application()
app.render_app()

#probabilities.insert(1, "image", [path_to_image_html("C:/Users/prest/Desktop/CS345/marvel-dialogue-nlp/tony-stark.jpg")] * 9)
#st.write(probabilities.to_html(escape=False, formatters=dict(column_name_with_image_links="./tony_stark_pic.jpg")))
#st.dataframe(probabilities.style.apply(custom_style, axis=1))

#st.write(probabilities.to_html(escape=False, index=False), unsafe_allow_html=True)

#html = probabilities.style.apply(custom_style).render()

#st.write(html, unsafe_allow_html=True)