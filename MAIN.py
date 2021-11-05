import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
import sys
class AI(QMainWindow):
    def __init__(self):
        super(AI, self).__init__()
        loadUi('AI.ui', self)

        self.cosine_similar_btn.clicked.connect(self.change_page_to_cosine)
        self.similar_btn.clicked.connect(self.find_cosine)

        self.back_from_cosine.clicked.connect(self.back_main)
        self.back_from_ml1.clicked.connect(self.back_main)
        self.back_from_ml2.clicked.connect(self.back_main)

        self.ML1_btn.clicked.connect(self.ml_mod1)
        self.predict_1.clicked.connect(self.one_predict)

        self.predict_2.clicked.connect(self.two_predict)
        self.ML2_btn.clicked.connect(self.ml_mod2)

        self.predict_3.clicked.connect(self.predict_temp)

    def predict_temp(self):
        temp = int(self.temperature.value())
        humi = int(self.humidity.value())

        df = pd.read_csv('weatherAUS.csv')
        df = df[['MaxTemp', 'Rainfall', 'RainToday', 'Humidity3pm']]

        dummy = pd.get_dummies(df['RainToday'])

        merged_df = pd.concat([df, dummy], axis='columns')

        merged_df = merged_df.fillna(0)

        model = LinearRegression()
        model.fit(merged_df[['MaxTemp', 'Humidity3pm']], merged_df[['Rainfall']])

        model2 = LogisticRegression()
        model2.fit(merged_df[['MaxTemp', 'Humidity3pm']], merged_df['Yes'])

        ans = model.predict([[temp, humi]])
        ans = ans[0][0]

        ans2 = model2.predict([[temp, humi]])
        ans2=ans2[0]

        if ans2 == 0:
            rain = 'No'
        else:
            rain = 'Yes'

        if temp > 45:
            self.info.setText("*Invalid Temperature")
        else:
            self.info.setText("")

            ans = round(ans, 2)

            if ans < 0:
                ans = 0

            if rain == 'No':
                self.will_it_rain.setText(str(rain))
            else:
                self.will_it_rain.setText(f"{str(rain)},  {ans}  mm")

    def ml_mod2(self):
        self.stackedWidget.setCurrentWidget(self.ml_mod_other)

    def two_predict(self):
        df = pd.read_csv('StudentsPerformance.csv')
        df_raw = df.drop(['parental level of education', 'test preparation course', 'lunch'], axis='columns')

        dummy = pd.get_dummies(df_raw['gender'])
        merged = pd.concat([df_raw, dummy], axis='columns')

        merged_df = merged.drop(['gender'], axis='columns')

        x = merged_df[['math score', 'writing score']]
        y = merged_df['female']

        model2 = LogisticRegression()
        model2.fit(x, y)

        ans = model2.predict([[self.math_score.value(), self.writing_score_2.value()]])

        if ans[0] == 0:
            gender = 'Male'
        else:
            gender = 'Female'

        self.gender_mod_first.setText(gender)

    def one_predict(self):
        df = pd.read_csv('StudentsPerformance.csv')
        df_raw = df.drop(['parental level of education', 'test preparation course', 'lunch'], axis='columns')

        dummy = pd.get_dummies(df_raw['gender'])
        merged = pd.concat([df_raw, dummy], axis='columns')

        merged_df = merged.drop(['gender'], axis='columns')

        X = merged_df[['female', 'writing score']]
        Y = merged_df[['math score']]

        model = LinearRegression()
        model.fit(X, Y)

        if self.gender.currentText() == 'Male':
            gender = 0
        else:
            gender = 1

        ans = model.predict([[gender, int(self.writing_score_1.value())]])

        if ans[0][0] < 0:
            self.label_24.setText(str(0))
        elif ans[0][0] > 100:
            self.label_24.setText(str(100))
        else:
            self.label_24.setText(str(round(ans[0][0], 2)))

    def ml_mod1(self):
        self.stackedWidget.setCurrentWidget(self.ml_mod_student)

        df = pd.read_csv('StudentsPerformance.csv')
        df_raw = df.drop(['parental level of education', 'test preparation course', 'lunch'], axis='columns')

        dummy = pd.get_dummies(df_raw['gender'])
        merged = pd.concat([df_raw, dummy], axis='columns')

        merged_df = merged.drop(['gender'], axis='columns')

        X = merged_df[['female', 'writing score']]
        Y = merged_df['math score']

        model = LinearRegression()
        model.fit(X, Y)
        self.progressBar.setValue((model.score(X, Y)) * 100)

        x = merged_df[['math score', 'writing score']]
        y = merged_df['female']

        model2 = LogisticRegression()
        model2.fit(x, y)
        self.progressBar_2.setValue((model2.score(x, y)) * 100)

    def back_main(self):
        self.stackedWidget.setCurrentWidget(self.main_page)

    def find_cosine(self):

        Sentence_1 = self.sent_1.text()
        Question = self.sent_2.text()

        Sentence_1 = Sentence_1.lower()
        Question = Question.lower()

        Vocabulary = []

        Sentence_1 = Sentence_1.split(' ')
        Question = Question.split(' ')

        for word in Sentence_1:
            if word not in Vocabulary:
                Vocabulary.append(word)

        for word in Question:
            if word not in Vocabulary:
                Vocabulary.append(word)

        for word in Vocabulary:
            word.lower()

        vocab = dict.fromkeys([i for i in Vocabulary], 0)
        vocab_3 = dict.fromkeys([i for i in Vocabulary], 0)

        for word in vocab:
            if word in Sentence_1:
                vocab[word] += 1
            else:
                vocab[word] = 0

        for word in vocab_3:
            if word in Question:
                vocab_3[word] += 1
            else:
                vocab_3[word] = 0

        bag_of_words = np.array([list(vocab.values()), list(vocab_3.values())])
        similar = cosine_similarity([bag_of_words[0]], [bag_of_words[-1]])

        value = similar[0][0]

        if value > 0.8:
            if self.sent_1.text().strip() == '' or self.sent_2.text().strip() == '':
                self.similar.setText('*Invalid')
            else:
                self.similar.setText("Very Similar")

        if 0.5 < value < 0.8:
            if self.sent_1.text().strip() == '' or self.sent_2.text().strip() == '':
                self.similar.setText('*Invalid')
            else:
                self.similar.setText("Moderately similar")

        if value < 0.5:
            if self.sent_1.text().strip() == '' or self.sent_2.text().strip() == '':
                self.similar.setText('*Invalid')
            else:
                self.similar.setText("Not similar")

    def change_page_to_cosine(self):
        self.stackedWidget.setCurrentWidget(self.cosine_page)

app = QApplication(sys.argv)
window = AI()
window.show()
app.exec_()
