from IPython import embed
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
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


    # Fit the Naive Bayes classifier with sklearn MultinomialNB() Model
    class_prior_vec = [1/3, 1/3, 1/3] # prior wird manuell gesetzt. Jedes Label gleich wahrscheinlich
    mnb = MultinomialNB( alpha = 0.01, fit_prior = False, class_prior = class_prior_vec)
    # Hat keine Verbesserung gebracht
    # mnb = MultinomialNB() ist die standard Funktion
    mnb.fit(X, y)

    print("parameters:" , mnb.get_params()) # 

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = mnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z[:, 1].reshape(xx.shape)
    Z2 = Z[:, 2].reshape(xx.shape)

    # print("Z:" , Z) # [0.33333374 0.33333273 0.33333353]
    # print("Z1:" , Z1)
    # print("Z2:" , Z2)


    classification = mnb.predict(X) 
    print(classification) # [0 0 2 ... 2 0 2]

    print("Accurancy:" , metrics.accuracy_score(y,classification))
    # Accurancy: 0.33366666666666667
    # macht fast alles in zwei Klassen und 50:50 recall weil er auf der falschen Achse klassifiziert

    print("Balanced accuracy:" , metrics.balanced_accuracy_score(y,classification))
    # Balanced accuracy: 0.33366666666666667

    print("Classification report:" , metrics.classification_report(y,classification))
    # siehe PNG in Github
  
    # Plot
def gen_plot():    
    fig = plt.figure(figsize=(5, 3.75))
    ax = fig.add_subplot(111)
    color_list = ['red','green','orange'] # red = class 0 , green = class 1, orange = class 2
    
    # Mit Class Punkten c=y
    plot1 = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colors.ListedColormap(color_list),
               edgecolor='k')

    # Mit classification
    # plot1 = ax.scatter(X[:, 0], X[:, 1], c=classification, cmap=colors.ListedColormap(color_list),
    #           edgecolor='k')

    ax.contour(xx,yy, -Z1, [-0.5],  colors='k')
    ax.contour(xx,yy, -Z2, [-0.5], colors='k')


    # Plot formatting
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

summary_op = tf.summary.image("MultinomialNB data 3000", image)

with tf.Session() as sess:
    # Run
    summary = sess.run(summary_op)
    # Write summary
    writer = tf.summary.FileWriter('./logs')
    writer.add_summary(summary)
    writer.close()