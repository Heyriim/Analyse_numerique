#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:54:16 2018

@author: canevean
"""

import numpy

# Question 1

#m1 = numpy.arange( 0, 14, 3)
#
#m2 = numpy.arange( 2, 15, 4)
#
#m3 = numpy.arange( 3, 16, 4)
#
#m4 = numpy.arange( 4, 17, 4)
#
#m4[3] = 0
#
#m = [ m1, m2, m3, m4]
#print(m)

#  Question 2

#A=[-2,-2,-2,-2,-2,-2,-2,-2,-2,-2]
#A1=[ 1, 1, 1, 1, 1, 1, 1, 1, 1] # numpy.ones((1,9))
#
#M=numpy.diag( A)  #M=-2*numpy.eye(10,10)
#M1=numpy.diag( A1, 1)
#M11=numpy.diag( A1, -1)
#
#print (M + M1 + M11)

#unM=numpy.ones(9)    
#M=-2*numpy.eye(10,10)+numpy.diag( unM, 1)+numpy.diag( unM, -1)
#print( M )
#
## Question 3
#
#def mat( n ):
#    mn = numpy.arange( 1, n**2+1)
#    mn2 = mn.reshape( n, n)
#    mn2 = mn2.T
#    mn2[ -1, -1] = 0
#    return mn2
#    
## Question 4 
#    
#def diagM(a,b,n):
#
#    unM=b*numpy.ones(n-1)    
#    M=a*numpy.eye(n,n)+numpy.diag( unM, 1)+numpy.diag( unM, -1)
#    print( M )
#
#diagM(-2,1,10)

# Question 5

A = numpy.array( [[0, 2, -1], [ 3, -2, 0], [ -2, 2, 1]])

D, P = numpy.linalg.eig(A) #Le premier argument représente les valeurs propres, tandis que le deuxième retourne la matrice des vecteurs propres en colonne (une colonne est un vecteur propre).
#print(ValeursPropres)

def Xn( X0, n) :
    Xn_ = A.dot(X0)
    for i in range( 1, n ):
        Xn_ = numpy.dot( A, Xn_)
    return Xn_

#print( Xn( [1, 2, 2], 4))

def matrice( X0, n) :
    P_ = numpy.linalg.inv( P )
    D1 = numpy.dot( numpy.dot(P_, A), P)
    D2 = numpy.dot( numpy.dot(P_, A@A), P)
    print( D2 )
    
matrice( [1, 2, 2], 4)
    


