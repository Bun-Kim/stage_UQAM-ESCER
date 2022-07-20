#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:32:21 2022

@author: san
"""
import csv

with open("Foudre_2015-2021.csv","r") as f:
    reader = csv.reader (f, delimiter = ";")
    
import pandas as pd

df = pd.read_csv("Foudre_2015-2021.csv")
df = pd.DataFrame
    