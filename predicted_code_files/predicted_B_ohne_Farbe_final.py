from IPython import embed
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd

from sklearn.naive_bayes import BernoulliNB
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from openpyxl.workbook import Workbook
 
import io
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    
    # Lokation von der Datei
    loc = "./predicted_test_result.xlsx" # Datei vom neuronalen Netz
    
    # Benutze Panda 
    mydata = pd.read_excel(loc) 

    # Gib die Spaltennamen aus
    print("Spaltenname: ", mydata.columns) # ['filename', 'logo-name', 'x', 'y', 'color']

    # Encoding logo-name in 0,1,2 damit c=y im plot funktioniert
    labelencoder = LabelEncoder()
    mydata['logo-name'] = labelencoder.fit_transform(mydata['logo-name'])

    # ordne die Spalte logo-name mit dem Label y zu
    y = mydata['logo-name'] # label 

    # Gib das Label/ Class aus
    print("Label: ", y) # Label:  0 2 usw. 
    # 0 = grey = hook, 1 = pink = triangle ,2 = orange = windows,

    # Nimm die zwei Features x und y jedoch noch nicht die Farbe
    selected_data = ['x','y']
    X = mydata[selected_data] # Featur x und y Koordinate

    # Da die Datei mit dp.read_excel() benutz müssen die Daten als Zahl konvertiert werden
    imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean')
    imputer = imputer.fit(X.iloc[:, :])
    X = imputer.transform(X.iloc[:, :])

    # Gebe die Features aus
    print("Features: ", X) # Features:  [32. 55.] usw.

    # für den Plot wichtig. Zeigt wie groß das Grid ist. Wir auf 8 gesetzt
    x_min, x_max = X[:, 0].min() - 8, X[:, 0].max() + 8
    y_min, y_max = X[:, 1].min() - 8, X[:, 1].max() + 8

    # Gib die x_min und die y_min Werte aus
    print("x_min:" , x_min) # x_min: -49.0
    print("y_min:" , y_min) # y_min: -19.0

    # Fit das Naive Bayes classifier mit dem sklearn BernoulliNB() Model
    class_prior_vec = [1/3, 1/3, 1/3] # prior wird manuell gesetzt. Jedes Label gleich wahrscheinlich
    bnb = BernoulliNB(binarize = 64.0, class_prior = class_prior_vec)
    # Binarize da es sich um die Bernoulli Verteilung handelt
    # Dies bringt eine deutliche Verbesserung 
    # 64 wird gewählt, da dies 128/2 ergibt
    # Ist in GitHub als improved hervorgehoben

    # bnb = BernoulliNB() 
    # Ist die standard Funktion
    # Beim ausführen kommt: warnings.warn("No contour levels were found")

    # Fit Naive Bayes classifier gemäß X,y
    bnb.fit(X, y)

    # Ein Raster von Punkten klassifizieren
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                         np.linspace(y_min, y_max, 30))
    Z = bnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # Errechnet die Wahrscheinlichkeitswerte für die Boundaries
    Z1 = Z[:, 1].reshape(xx.shape) # setzt zwei Boundaries
    Z2 = Z[:, 2].reshape(xx.shape) # setzt zwei Boundaries

    #print("Z:" , Z) 
    #print("Z1:" , Z1)
    #print("Z2:" , Z2)


    # Classification als eine Variable abspeichern und ausgeben
    classification = bnb.predict(X)
    print(classification) # [0 0 0 ... 0 0 0]
    
    print("Accuracy:" , metrics.accuracy_score(y, classification))
    # Accuracy: 0.339
    # macht fast alles in eine Klasse und dadurch ca. 1/3 richtig
    # Accuracy improved: 0.7743333333333333

    print("Balanced accuracy:" , metrics.balanced_accuracy_score(y,classification))
    # Balanced accuracy: 0.33899999999999997
    # Balanced accuracy improved: 0.7743333333333333
    # Gleiches Ergebnis da selbe Anzahl an Datenpunken. 3 Labels jeweils 1000 Punkte

    # Print
    print("Classification report:" , metrics.classification_report(y,classification))
    # siehe PNG in GitHub

    # Hier beginnt der Plot
def gen_plot():    
    fig = plt.figure(figsize=(5, 3.75))
    ax = fig.add_subplot(111)
    color_list = ['grey','pink','orange'] # grey = class 0 , pink = class 1, orange = class 2
    
    # Ohne Classification c=y
    # plot1 = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colors.ListedColormap(color_list),
    #          edgecolor='k')

    # Mit Classification
    plot1 = ax.scatter(X[:, 0], X[:, 1], c=classification, cmap=colors.ListedColormap(color_list),
              edgecolor='k')


    # c=y zeigt die Punkte mit dem richtigen Label
    # c=classification zeigt die Punkte eingeteilt durch die decision boundray

    # Zeichen vertauschen, damit die Kontur gestrichelt wird (MPL default)
    ax.contour(xx, yy, -Z1,[-0.5], colors='k')
    ax.contour(xx, yy, -Z2,[-0.5], colors='k')

    # Plot Formatierung 
    ax.set_xlabel('x-Werte')
    ax.set_ylabel('y-Werte')
    ax.set_title('sklearn decision boundary BernoulliNB')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    plt.tight_layout()

    # Für Tensorboard wichtig
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Das Bild der Darstellung ist als PNG in Github hinterlegt

# Bereite den Plot vor
plot_buf = gen_plot()

# PNG-Puffer in TF-Bild konvertieren
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

# Hinzufügen der Batch-Dimension
image = tf.expand_dims(image, 0)

# Bildübersicht hinzufügen
summary_op = tf.summary.image("BernoulliNB with data from the neural network", image)

# Starte die Session
with tf.Session() as sess:
    # Run
    summary = sess.run(summary_op)
    # Schreibe die summary
    writer = tf.summary.FileWriter('./logs')
    writer.add_summary(summary)
    writer.close()