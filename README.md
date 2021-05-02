# Project 2: Mini deep learning framework

## Modules

Module c la superclass, méthodes *forward*, *backward*, parametres *parameters* (list de tuples (parameter, grad) )
-> Linear (ou Layer, mais c'est la seule pour l'instant) aura la spécificité d'avoir une backward qui update ses poids
-> Relu doit avoir une backward qui output un gradient soit 0 soit le gradient / Tanh pareil

## Utilities

LossMSE compute une loss et renvoie Tensor[value, gradient]
class optimizer(model.params, lr, momentum) qui utilise cette loss pour update les poids avec la méthode *step()* (une SGD)

## Network
class Sequential(*args) qui combine plusieurs modules 

## Data
Le générateur de data
