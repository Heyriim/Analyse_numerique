# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:21:15 2018

@author: piganega
"""

import numpy as np
import matplotlib.pyplot as plt

#Equation d'evolution

def dynamique(t,x):
    a=1
    return -a*x 

def f(t,x) : # x[0]=x(t) et x[1]=y(t) dans le vecteur d'état
    a=4
    b=3
    c=3
    d=2
    xdot=x[0]*(a-b*x[1])
    ydot=-x[1]*(c-d*x[0])
    
    return np.array([xdot,ydot])

def euler(T, tmax, h, y0, dynamique) :
    
    sol=[y0]                  # liste pour stocker les résultats
    
    for t in T[1:] : 
        
        y0 = y0 + h*dynamique(t,y0)
        sol.append(y0)
        
    sol_np = np.array(sol)
    
    return sol_np

def RungeKutta2(T,tmax,h,etat,dynamique):
    
    lambda1=2/3
    c2=1/(2*lambda1)
    c1=1-c2
    sol=[etat]                  # liste pour stocker les résultats
    
    for t in T[1:] : 
        f1 =dynamique(t,etat)
        f2=dynamique(t+lambda1*h,etat+lambda1*h*f1)
        etat=etat+h*(c1*f1+c2*f2)
        sol.append(etat)
    sol_np = np.array(sol)
    
    return sol_np

def RungeKutta3(T,tmax,h,etat,f):
    
    lambda1=2/3
    sol=[etat]                  # liste pour stocker les résultats
    
    for t in T[1:] : 
        k1 =f(t,etat)
        k2=f(t+1/3*h,etat+1/3*h*k1)
        k3=f(t+lambda1*h,etat+lambda1*h*k2)
        etat=etat+h/4*(k1+3*k3)
        sol.append(etat)
    sol_np = np.array(sol)
    
    return sol_np


if __name__ == "__main__":
    
    choix = 4
    if choix == 4 :
        # ========= Condition Initial ========= # 
        
        etat=np.array([2, 0.25])    # condition initiale
        h= 0.05                  # pas d'untégration 
        tmax = 15    
        T = np.arange(0,tmax,h)
        
        # ==================================== # 
        
        points= RungeKutta3(T,tmax,h,etat,f)
        
         # Fenêtre 1 
        
        plt.figure(1)
        
        # Méthode RK3
        
        plt.subplot(221)
        plt.plot(T,points)
        plt.title("RK3 - Proies - Evolution")
        plt.ylabel("Nombre")
        plt.xlabel("Temps")
        
        plt.subplot(222)
        plt.plot(points[:,0],points[:,1])
        plt.title("RK3 - Proies/Prédateurs")
        plt.xlabel("Proies")
        plt.ylabel("Predateurs")

        points2=RungeKutta2(T,tmax,h,etat,f)
        
        # Fenêtre 1 
        
        plt.figure(1)
        
        # Méthode RK2
        
        plt.subplot(223)
        plt.plot(T,points2)
        plt.title("RK2 - Proies - Evolution")
        plt.ylabel("Nombre")
        plt.xlabel("Temps")
        
        plt.subplot(224)
        plt.plot(points2[:,0],points2[:,1])
        plt.title("RK2 - Proies/Prédateurs")
        plt.xlabel("Proies")
        plt.ylabel("Predateurs")

    if choix == 3 :
        # ========= Condition Initial ========= # 
        
        etat=np.array([2, 0.25])    # condition initiale
        h= 0.001                    # pas d'untégration 
        tmax = 10    
        T = np.arange(0,tmax,h)
        
        # ==================================== # 
        points=RungeKutta2(T,tmax,h,etat,f)
        
        # Fenêtre 1 
        
        plt.figure(1)
        
        # Méthode RK2
        
        plt.subplot(221)
        plt.plot(T,points)
        plt.title("RK2 - Proies - Evolution")
        plt.ylabel("Nombre")
        plt.xlabel("Temps")
        
        plt.subplot(222)
        plt.plot(points[:,0],points[:,1])
        plt.title("RK2 - Proies/Prédateurs")
        plt.xlabel("Proies")
        plt.ylabel("Predateurs")
    
    if choix == 2 : 
        
        # ========= Condition Initial ========= # 
        
        etat=np.array([2, 0.25])    # condition initiale
        h= 0.01                     # pas d'untégration 
        tmax = 10    
        T = np.arange(0,tmax,h)
        
        # ==================================== # 
    
        points=euler(T,tmax,h,etat,f)
        
        # Fenêtre 1 
        
        plt.figure(1)
        
        # Méthode Euler
        
        plt.subplot(221)
        plt.plot(T,points)
        plt.title("Euler - Proies - Evolution")
        plt.ylabel("Nombre")
        plt.xlabel("Temps")
        
        plt.subplot(222)
        plt.plot(points[:,0],points[:,1])
        plt.title("Euler - Proies/Prédateurs")
        plt.xlabel("Proies")
        plt.ylabel("Predateurs")
 

       
    if choix == 1 : 
        # ========= Condition Initial ========= # 
        
        etat=np.array([1])    # condition initiale
        # sol=[etat]          # liste pour stocker les résultats
        h= 1*np.exp(-3)       # pas d'untégration 
        tmax = 10    
        T = np.arange(0,tmax,h)
        
        # ==================================== #
            
        points = euler(T,tmax,h,etat,dynamique)
        
        # Fenêtre 1 
        
        plt.figure(1)
        
        # Méthode Euler
        
        plt.subplot(121)
        plt.plot(T,points)
        plt.title("Approximation par Euler")
        plt.xlabel("Temps")
        
        # Comparer les valeurs : erreur
        
        plt.subplot(122)
        erreur=np.abs(points.flatten()-np.exp(-T))
        plt.plot(T,erreur)
        plt.title("Erreur")
        plt.xlabel("Temps")