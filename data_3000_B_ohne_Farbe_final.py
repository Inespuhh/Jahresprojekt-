from IPython import embed
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd

from sklearn.naive_bayes import BernoulliNB
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
    print("Label: ", y) # z.B. 0 0
    
    selected_data = ['x','y']
    X = mydata[selected_data] # Featur x und y Koordinate

    # Featuers
    imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean')
    imputer = imputer.fit(X.iloc[:, :])
    X = imputer.transform(X.iloc[:, :])

    print("Features: ", X) # z.B. [ 80.  68.]

    x_min, x_max = X[:, 0].min() - 8, X[:, 0].max() + 8 
    y_min, y_max = X[:, 1].min() - 8, X[:, 1].max() + 8

    print("x_min:" , x_min) 
    print("y_min:" , y_min) 

    # Fit the Naive Bayes classifier with sklearn BernoulliNB() Model
    class_prior_vec = [1/3, 1/3, 1/3] # prior jede classe gleich wahrscheinlich
    bnb = BernoulliNB(binarize = 64.0, class_prior = class_prior_vec)# binarize da Bernoulli und class_prior selbst gesetzt
    # Hat eine deutliche Verbesserung gebracht
    # bnb = BernoulliNB()
    bnb.fit(X, y)

    # Classify a grid of points
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

    print("Accurancy:" , metrics.accuracy_score(y,classification))
    # Accurancy: 0.793
    # macht fast alles in eine Klasse und dadurch 1/3 richtig

    print("Balanced accuracy:" , metrics.balanced_accuracy_score(y,classification))
    # Balanced accuracy: 0.7930000000000001 
    # selbe anzahl an Datenpunken.

    print("Classification report:" , metrics.classification_report(y,classification))
    # siehe Word snipping 

    # Plot
def gen_plot():    
    fig = plt.figure(figsize=(5, 3.75))
    ax = fig.add_subplot(111)
    color_list = ['red','green','orange'] # red = class 0 , green = class 1, orange = class 2
    
    # Mit Class Punkten c=y
    plot1 = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colors.ListedColormap(color_list),
               edgecolor='k')

    # Mit Classification 
    # plot1 = ax.scatter(X[:, 0], X[:, 1], c=classification, cmap=colors.ListedColormap(color_list),
    #            edgecolor='k')

    # Swap signs to make the contour dashed (MPL default)
    ax.contour(xx, yy, -Z1,[-0.5], colors='k')
    ax.contour(xx, yy, -Z2,[-0.5], colors='k')

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
plot_buf = gen_plot()

# Convert PNG buffer to TF image
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

# Add the batch dimension
image = tf.expand_dims(image, 0)

# Add image summary
summary_op = tf.summary.image("BernoulliNB data 3000", image)

# Session
with tf.Session() as sess:
    # Run
    summary = sess.run(summary_op)
    # Write summary
    writer = tf.summary.FileWriter('./logs')
    writer.add_summary(summary)
    writer.close()