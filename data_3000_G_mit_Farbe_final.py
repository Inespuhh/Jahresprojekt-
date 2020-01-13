from IPython import embed
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn import metrics
# import xlrd # reading excel file
# from xlrd import open_workbook 
import io
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    
    # Location of file
    loc = "./data3000.xlsx"
    # Use Panda 
    mydata = pd.read_excel(loc) 

    # Print data 
    print("Spaltenname: ", mydata.columns)

    # take two features without color
    y = mydata['logo-name_codiert'] # label 

    # Print Label/ Class
    # print("Label: ", y) # z.B. 0 0
    
    selected_data = ['x','y','farbe_codiert']
    X = mydata[selected_data] # Features x und y Koordinate mit Farbe

    # Featuers
    imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean')
    imputer = imputer.fit(X.iloc[:, :])
    X = imputer.transform(X.iloc[:, :])

    print("Features: ", X) # z.B. [ 80.  68.   0.]
    # 0 = red, 1 = green und 2 = blue. Farbe wird als Störfaktor eingesetzt. 

    # für den Plot, wie groß das Grid ist, auf 8 gesetzt
    x_min, x_max = X[:, 0].min() - 8, X[:, 0].max() + 8
    y_min, y_max = X[:, 1].min() - 8, X[:, 1].max() + 8
    z_min, z_max = X[:, 2].min() - 8, X[:, 2].max() + 8

    print("x_min:" , x_min) # -8 
    print("y_min:" , y_min) # -8 
    print("z_min:" , z_min) # -8

    # Fit the Naive Bayes classifier with sklearn GaussianNB() Model
    gnb = GaussianNB()
    gnb.fit(X, y)

    # Classify a grid of points
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 30),
                             np.linspace(y_min, y_max, 30),
                             np.linspace(z_min, z_max, 30))
   
    Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    
    Z1 = Z[:, 1].reshape(xx.shape) # setzt zwei Boundaries
    Z2 = Z[:, 2].reshape(xx.shape) # setzt zwei Boundaries

    print("Z:" , Z) # [3.94670368e-05 1.30883416e-14 9.99960533e-01]
    #print("Z1:" , Z1)
    #print("Z2:" , Z2)


    # Für Marc mit predict(self, X) ausgeben. X, Y und classification in excel speichern
    classification = gnb.predict(X)
    print(classification) # [1 1 0 ... 0 2 1]

    print("Accurancy:" , metrics.accuracy_score(y, classification))
    # Accurancy: 0.9176666666666666 mit Farbe
    # Accurancy: 0.9183333333333333 ohne Farbe
    # Die Farbe hilft nicht und macht es nicht schlechter. Das Feature wird vom Algorithmus nicht mit rein genommen was gut ist. 

    print("Balanced accuracy:" , metrics.balanced_accuracy_score(y,classification))
    # Balanced accuracy: 0.9176666666666667
    # selbe anzahl an Datenpunken. Jeweils 1000 Punkte. 

    print("Classification report:" , metrics.classification_report(y,classification))
    # siehe Word snipping 

    # Plot
def gen_plot():    
    fig = plt.figure(figsize=(5, 3.75))
    ax = fig.add_subplot(111)
    color_list = ['red','green','orange'] # red = class 0 , green = class 1, orange = class 2

    # Ohne Classification c=y
    plot1 = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colors.ListedColormap(color_list),
                edgecolor='k')
    # X[:,2]
    
    # Mit Classification c=y ersetzt mit c=classification
    # plot1 = ax.scatter(X[:, 0], X[:, 1], c=classification, cmap=colors.ListedColormap(color_list),
    #            edgecolor='k')

    # c=y zeigt die Punkte mit dem richtigen Label
    # c=classification zeigt die Punkte eingeteilt durch die decision boundray

    # Swap signs to make the contour dashed (MPL default)
    ax.contour(xx, yy, -Z1,[-0.5], colors='k')
    ax.contour(xx, yy, -Z2,[-0.5], colors='k')

    # ax.contour(xx, yy, zz, -Z1,[-0.5], colors='k')
    # ax.contour(xx, yy, zz, -Z2,[-0.5], colors='k')


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

# Prepare the plot
plot_buf = gen_plot() # TypeError: Input z must be a 2D array.

# Convert PNG buffer to TF image
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

# Add the batch dimension
image = tf.expand_dims(image, 0)

# Add image summary
summary_op = tf.summary.image("GaussianNB data 3000", image)

# Session
with tf.Session() as sess:
    # Run
    summary = sess.run(summary_op)
    # Write summary
    writer = tf.summary.FileWriter('./logs')
    writer.add_summary(summary)
    writer.close()