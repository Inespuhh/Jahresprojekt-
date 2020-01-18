from IPython import embed
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn import metrics
import io
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    
    # Lokation von der Datei
    loc = "./data3000.xlsx"

    # Benutze Panda 
    mydata = pd.read_excel(loc) 

    # Gib die Spaltennamen aus
    print("Spaltenname: ", mydata.columns)

    # ordne die Spalte logo-name_codiert mit dem Label y zu
    # Hier wurde das Label in der Excel codiert
    y = mydata['logo-name_codiert'] # label 

    # Gib das Label/ Class aus
    # print("Label: ", y) # z.B. 0 0
    
    # Nimm die zwei Features x und y jedoch noch nicht die Farbe
    selected_data = ['x','y']
    X = mydata[selected_data] # Featur x und y Koordinate

    # Da die Datei mit dp.read_excel() benutz müssen die Daten als Zahl konvertiert werden
    imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean')
    imputer = imputer.fit(X.iloc[:, :])
    X = imputer.transform(X.iloc[:, :])

    # Gebe die Features aus
    print("Features: ", X) # z.B. [ 80.  68.]

    # für den Plot wichtig. Zeigt wie groß das Grid ist. Wir auf 8 gesetzt
    x_min, x_max = X[:, 0].min() - 8, X[:, 0].max() + 8
    y_min, y_max = X[:, 1].min() - 8, X[:, 1].max() + 8
    # Rechne bei den Achsen von 0 bis 128 jeweils minus 8 und plus 8 drauf. 
    # X-Achse startet bei -8 und endet bei 130.
    # Damit die Punkte nicht auf den Linien der Achsen gezeigt werden.
    # Siehe Bild GaussianNB_data_3000_with_classification_with_axes_values.PNG 

    # Gib die x_min und die y_min Werte aus
    print("x_min:" , x_min) 
    print("y_min:" , y_min) 

    # Fit das Naive Bayes classifier mit dem sklearn GaussianNB() Model
    gnb = GaussianNB()
    gnb.fit(X, y)

    # Ein Raster von Punkten klassifizieren
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                         np.linspace(y_min, y_max, 30))

    # # Errechnet die Wahrscheinlichkeitswerte für die Boundaries
    Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    Z1 = Z[:, 1].reshape(xx.shape) # setzt zwei Boundaries
    Z2 = Z[:, 2].reshape(xx.shape) # setzt zwei Boundaries
    # Nur zwei Boundaries da es drei Labels sind

    # print("Z:" , Z) # [6.42166085e-07 2.65583776e-13 9.99999358e-01] 
    # 9.9 -> sehr hohe Wahrscheinlichkeit das der Punkt zu dem einem label gehört.
    #print("Z1:" , Z1)
    #print("Z2:" , Z2)


    # Classification als eine Variable abspeichern und ausgeben
    classification = gnb.predict(X)
    print(classification) # [0 0 0 ... 0 2 1]

    # Marc CSV und dann in Excel
    
    # CSV Datei Vorbeireitung
    # print(X)
    select_for_csv = np.empty((len(classification),3))
    select_for_csv[:,0] = X[:,0]
    select_for_csv[:,1] = X[:,1]
    select_for_csv[:,2] = classification[:].transpose()

    # Gib die Vorbereitung für CSV aus
    print("Daten für CSV: ", select_for_csv)

    # Dataframe Inhalte angeben und danach ausgeben
    df = pd.DataFrame(select_for_csv)
    print("DataFrame: ", df)

    # Die Daten in eine CSV Datei schreiben
    df.to_csv("./data_Marc.csv", sep='\t')

    print("Accurancy:" , metrics.accuracy_score(y, classification))
    # Accurancy: 0.9183333333333333
    # 90 % richtig zugeordnet und die 10% random durch Zufall eingeordnet

    print("Balanced accuracy:" , metrics.balanced_accuracy_score(y,classification))
    # Balanced accuracy: 0.9183333333333333
    # Gleiches Ergebnis da selbe Anzahl an Datenpunken. 3 Labels jeweils 1000 Punkte

    print("Classification report:" , metrics.classification_report(y,classification))
    # siehe PNG in GitHub

    # Hier beginnt der Plot
def gen_plot():    
    fig = plt.figure(figsize=(5, 3.75))
    ax = fig.add_subplot(111)
    color_list = ['red','green','orange'] # red = class 0 , green = class 1, orange = class 2
    
    # Ohne Classification c=y
    # plot1 = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colors.ListedColormap(color_list),
    #             edgecolor='k')

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
    ax.set_title('sklearn decision boundary data 3000')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_xticks(())
    # ax.set_yticks(())
    ax.set_xticks((x_min, x_max))
    ax.set_yticks((y_min, y_max))

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
summary_op = tf.summary.image("GaussianNB data 3000", image)

# Starte die Session
with tf.Session() as sess:
    # Run
    summary = sess.run(summary_op)
    # Write summary
    writer = tf.summary.FileWriter('./logs')
    writer.add_summary(summary)
    writer.close()