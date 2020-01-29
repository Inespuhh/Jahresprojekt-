from IPython import embed
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd

from sklearn.naive_bayes import BernoulliNB
from sklearn.impute import SimpleImputer
from sklearn import metrics
import io
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    
    loc = "./data3000.xlsx"
   
    mydata = pd.read_excel(loc) 

    print("Spaltenname: ", mydata.columns)

    y = mydata['logo-name_codiert'] # label 

    print("Label: ", y) # z.B. 0 0
    
    selected_data = ['x','y']
    X = mydata[selected_data] # Featur x und y Koordinate

    imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean')
    imputer = imputer.fit(X.iloc[:, :])
    X = imputer.transform(X.iloc[:, :])

    print("Features: ", X) # z.B. [ 80.  68.]

    x_min, x_max = X[:, 0].min() - 8, X[:, 0].max() + 8 
    y_min, y_max = X[:, 1].min() - 8, X[:, 1].max() + 8

    print("x_min:" , x_min) 
    print("y_min:" , y_min) 

    # Fit the Naive Bayes classifier with sklearn BernoulliNB() Model
    class_prior_vec = [1/3, 1/3, 1/3] # prior wird manuell gesetzt. Jedes Label gleich wahrscheinlich
    bnb = BernoulliNB(binarize = 64.0, class_prior = class_prior_vec)
    # Binarize da es sich um die Bernoulli Verteilung handelt
    # Dies bringt eine deutliche Verbesserung 
    # 64 wird gewählt, da dies 128/2 ergibt
    # Ist in GitHub als improved hervorgehoben

    # bnb = BernoulliNB() ist die standard Funktion
    bnb.fit(X, y)

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                         np.linspace(y_min, y_max, 30))
    Z = bnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z[:, 1].reshape(xx.shape)
    Z2 = Z[:, 2].reshape(xx.shape)

    # print("Z:" , Z) # [0.22222222 0.11111111 0.66666667]
    # print("Z1:" , Z1)
    # print("Z2:" , Z2)

    classification = bnb.predict(X) 
    print(classification) # [1 1 1 ... 0 0 1]

    print("Accuracy:" , metrics.accuracy_score(y,classification))
    # Accuracy: 0.793
    # macht fast alles in eine Klasse und dadurch 1/3 richtig

    print("Balanced accuracy:" , metrics.balanced_accuracy_score(y,classification))
    # Balanced accuracy: 0.7930000000000001 

    print("Classification report:" , metrics.classification_report(y,classification))
    # siehe PNG in GitHub 

    # Hier beginnt der Plot
def gen_plot():    
    fig = plt.figure(figsize=(5, 3.75))
    ax = fig.add_subplot(111)
    color_list = ['red','green','orange'] # red = class 0 , green = class 1, orange = class 2
    
    plot1 = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colors.ListedColormap(color_list),
               edgecolor='k')

    # Mit Classification 
    # plot1 = ax.scatter(X[:, 0], X[:, 1], c=classification, cmap=colors.ListedColormap(color_list),
    #            edgecolor='k')

    ax.contour(xx, yy, -Z1,[-0.5], colors='k')
    ax.contour(xx, yy, -Z2,[-0.5], colors='k')

    ax.set_xlabel('x-Werte')
    ax.set_ylabel('y-Werte')
    ax.set_title('sklearn decision boundary data 3000')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Das Bild der Darstellung ist als PNG in Github hinterlegt

plot_buf = gen_plot()

image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

image = tf.expand_dims(image, 0)

summary_op = tf.summary.image("BernoulliNB data 3000", image)

with tf.Session() as sess:
    # Run
    summary = sess.run(summary_op)
    # Write summary
    writer = tf.summary.FileWriter('./logs')
    writer.add_summary(summary)
    writer.close()