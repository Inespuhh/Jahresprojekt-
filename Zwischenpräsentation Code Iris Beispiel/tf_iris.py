#!/usr/bin/env python3

"""
Flower classification on the Iris dataset using a Naive Bayes
classifier and TensorFlow.

For more info: http://nicolovaligi.com/naive-bayes-tensorflow.html
"""

from IPython import embed
from matplotlib import colors
from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np
import tensorflow as tf
import tensorflow_probability  as tfp
from sklearn.utils.fixes import logsumexp
import numpy as np


class TFNaiveBayesClassifier:
    dist = None # Parameter wird auf none gesetzt

    def fit(self, X, y): # Decision boundary
        # Separate training points by class (nb_classes * nb_samples * nb_features)
        unique_y = np.unique(y) # y für jeden Datenpunkt welche Klasse, unique y 0,1,2 für die Klasse
        # Formatiert die Daten und trennt die Features in 2 Spalten auf. 
        # Die Features werden nach dem label sortiert.
        points_by_class = np.array([ 
            [x for x, t in zip(X, y) if t == c]
            for c in unique_y])

        # Schätzt mean und variance für jede Klasse / Feature
        # shape: nb_classes * nb_features
        mean, var = tf.nn.moments(tf.constant(points_by_class), axes=[1])  

        # Normalverteilung loc location der Spitze, scale breite der Kurve 
        # Create a 3x2 univariate normal distribution with the known mean and variance
        self.dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(var)) # Quadradwurzel von var

    def predict(self, X): 
        assert self.dist is not None # Test nur weiter mit der Funktion wenn etwas darin steht. 
        nb_classes, nb_features = map(int, self.dist.scale.shape) 
        # y will man herauskriegen. Die Frage ist was die Klasse/ Art der Pflanze ist. 
        # Mit den zwei Features soll man dies heraus bekommen.
        # Map kann als eine Rundungsfunktion eingesetzt werden und erzwinkt int Werte. 
        # nb classes kommt 3 raus, nb features sollte 2 sein. 

        # Conditional probabilities log P(x|c) with shape 
        # Wahrscheinlichkeit das Testpunkt x zu klasse c gehört. 
        # Bei einem neuen Dateneintrag soll gesagt werden mit welcher Wahrscheinlichkeit dieser Punkt welche Klasse hat. 
        # Das Ergebnis soll P(c|x) sein. 
        # Rechnet eine Matrix aus. Rechnet es für alle klassen die Wahrscheinlichkeit aus.
        # Aus 2 dim martix mache eine 1 dim vektor, summiert die Felder
        cond_probs = tf.reduce_sum( 
            self.dist.log_prob( 
                tf.reshape( 
                    tf.tile(X, [1, nb_classes]), [-1, nb_classes, nb_features])),
            axis=2) 

        # uniform priors. Vorherschende Glaube. 
        # Dies hängt von der Anzahl der klassen ab. 
        # Ist vereinfacht da alle Klassen die selbe wahrscheinlichkeit haben
        # teilt durch 1/3 weil alle die gleiche wahrscheinlichkeit haben 
        # Auch hier wird die log funktion benutzt
        priors = np.log(np.array([1. / nb_classes] * nb_classes)) 

        # posterior log probability, log P(c) + log P(x|c)
        # Bayes wird angewendet und prior und conditional prob wird verwendet um die likelihood auszurechnen 
        # log Funktion benutzt und kann deshalb als plus zusammen gezählt werden
        joint_likelihood = tf.add(priors, cond_probs) 
        
        # normalize to get (log)-probabilities
        # Will es wegen dem log zurückrechnen und normalisieren. 
        norm_factor = tf.reduce_logsumexp(
            joint_likelihood, axis=1, keepdims=True) 
        # Über die Testpunkte wird normalisiert
        
        log_prob = joint_likelihood - norm_factor 
       
        # Die exponetial Funktion wird benutzt, damit log weg fällt.
        # Bei jedem Testpunkt steht dran zu welcher Wahrscheinlichkeit die drei Klassen zu diesem gehören
        # exp to get the actual probabilities
        # Alle drei Prozentwerte müssen bei den Klassen insgesamt 100% geben. 
        return tf.exp(log_prob)  
        

if __name__ == '__main__':
    iris = datasets.load_iris()
    # Numm nur die ersten zwei Features
    X = iris.data[:, :2]  
    y = iris.target # label 

    tf_nb = TFNaiveBayesClassifier() # instanz der Klasse. Es wird kein Argument übergeben
    tf_nb.fit(X, y) #  Normalverteilung mean und variance.  
    
    # Create a regular grid and classify each point
    # Was ist der kleinste Punkt und ziehe davon noch etwas ab. 
    # Die Grafik hat dadurch mehr Platz zu den Seiten. 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), 
                         np.linspace(y_min, y_max, 30))  
    s = tf.Session()

    # Gib einen Punkt ein und es sagt zu welcher dieser zu einer Klasse gehört.
    Z = s.run(tf_nb.predict(np.c_[xx.ravel(), yy.ravel()])) 
    
    # Extract probabilities of class 2 and 3
    Z1 = Z[:, 1].reshape(xx.shape)
    Z2 = Z[:, 2].reshape(xx.shape)
    print(nb_classes)
    
    # Plot
    fig = plt.figure(figsize=(5, 3.75))
    ax = fig.add_subplot(111)

    # Die Testpunkte werden visualisiert.
    # Die Punkte werden in der Farbe der zugehörigen Klasse dargestellt.
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, 
                edgecolor='k')

    # Swap signs to make the contour dashed (MPL default)
    ax.contour(xx, yy, -Z1, [-0.5], colors='k') # gestrichelte Linie wird gezeichet
    ax.contour(xx, yy, -Z2, [-0.5], colors='k') # 2 Linien werden gezeichnet 

    # Bezeichnung der Grafik
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_title('TensorFlow decision boundary')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    
    # Speichere die Grafik
    plt.tight_layout()
    fig.savefig('tf_iris1.png', bbox_inches='tight')
