import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np

%matplotlib inline
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
print(np.shape(imagenes), n_imagenes) # Hay 1797 digitos representados en imagenes 8x8
# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Vamos a hacer un split training test
scaler = StandardScaler()
#StandardScaler resta la media y pone varianza cero.
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# Vamos a entrenar solamente con los digitos iguales a 1
numero = 1 #¿Cómo sabe que los datos son uno? Están etiquetados en el atributo target.
dd = y_train==numero
#matriz de covarianza de pca
cov = np.cov(x_train[dd].T)
#valores y vectores propios
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
#hay que elegir los valores más importantes (grandes)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

unos=np.mean(np.dot(x_train[dd],vectores[:,0]),axis=0)
desv=np.std(np.dot(x_train[dd],vectores[:,0],axis=0)
            
            
for i in range(6):
    unos=unos+np.mean(np.dot(x_train[dd],vectores[:,i]),axis=0)
    desv=desv+np.std(np.dot(x_train[dd],vectores[:,i],axis=0)
#ahora, el número debería estar en la media  en el espacio de los vectores propios de uno
ones=[]
for x in x_train:
    prom=np.mean(np.dot(x,vectores[:,0]),axis=0)
    for i in range(6):
        prom=prom+np.mean(np.dot(x,vectores[:,i]),axis=0)
    if (prom>unos-desv and prom<unos+desv):
        np.append(ones,1)
    else:
        np.append(ones,0)
print('Predict:',ones)

true_positive=0
