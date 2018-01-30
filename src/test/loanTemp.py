#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import csv


#readerTemp = pd.read_csv('loans.csv')
readerTemp = open('loans.csv', 'r', encoding='utf-8')
CsvReader = csv.DictReader(readerTemp)
List = pd.DataFrame()

i = 0
for rowTemp in CsvReader:
    #print(rowTemp['grade'])
    if rowTemp['grade'] == 'A':
        coe = 30
        if rowTemp['sub_grade'] == 'A1':
            grade = coe + 5
        elif rowTemp['sub_grade'] == 'A2':
            grade = coe + 4
        elif rowTemp['sub_grade'] == 'A3':
            grade = coe + 3
        elif rowTemp['sub_grade'] == 'A4':
            grade = coe + 2
        else:
            grade = coe + 1
            
    elif rowTemp['grade'] == 'B':
        coe = 25
        if rowTemp['sub_grade'] == 'B1':
            grade = coe + 5
        elif rowTemp['sub_grade'] == 'B2':
            grade = coe + 4
        elif rowTemp['sub_grade'] == 'B3':
            grade = coe + 3
        elif rowTemp['sub_grade'] == 'B4':
            grade = coe + 2
        else:
            grade = coe + 1 
            
    elif rowTemp['grade'] == 'C':
        coe = 20
        if rowTemp['sub_grade'] == 'C1':
            grade = coe + 5
        elif rowTemp['sub_grade'] == 'C2':
            grade = coe + 4
        elif rowTemp['sub_grade'] == 'C3':
            grade = coe + 3
        elif rowTemp['sub_grade'] == 'C4':
            grade = coe + 2
        else:
            grade = coe + 1
            
    elif rowTemp['grade'] == 'D':
        coe = 15
        if rowTemp['sub_grade'] == 'D1':
            grade = coe + 5
        elif rowTemp['sub_grade'] == 'D2':
            grade = coe + 4
        elif rowTemp['sub_grade'] == 'D3':
            grade = coe + 3
        elif rowTemp['sub_grade'] == 'D4':
            grade = coe + 2
        else:
            grade = coe + 1                 
            
            
    elif rowTemp['grade'] == 'E':
        coe = 10
        if rowTemp['sub_grade'] == 'E1':
            grade = coe + 5
        elif rowTemp['sub_grade'] == 'E2':
            grade = coe + 4
        elif rowTemp['sub_grade'] == 'E3':
            grade = coe + 3
        elif rowTemp['sub_grade'] == 'E4':
            grade = coe + 2
        else:
            grade = coe + 1 

    elif rowTemp['grade'] == 'F':
        coe = 5
        if rowTemp['sub_grade'] == 'F1':
            grade = coe + 5
        elif rowTemp['sub_grade'] == 'F2':
            grade = coe + 4
        elif rowTemp['sub_grade'] == 'F3':
            grade = coe + 3
        elif rowTemp['sub_grade'] == 'F4':
            grade = coe + 2
        else:
            grade = coe + 1
            
    else:
        coe = 0
        if rowTemp['sub_grade'] == 'G1':
            grade = coe + 5
        elif rowTemp['sub_grade'] == 'G2':
            grade = coe + 4
        elif rowTemp['sub_grade'] == 'G3':
            grade = coe + 3
        elif rowTemp['sub_grade'] == 'G4':
            grade = coe + 2
        else:
            grade = coe + 1
            
    #print('grade',grade)
    
    if rowTemp['home_ownership'] == 'RENT':
        home_ownership = 1
    elif rowTemp['home_ownership'] == 'MORTGAGE':
        home_ownership = 2
    else:
        home_ownership = 3
        
    #print('home_ownership',home_ownership)

    if rowTemp['purpose'] == 'car':
        purpose = 1
    elif rowTemp['purpose'] == 'credit_card':
        purpose = 2
    elif rowTemp['purpose'] == 'debt_consolidation':
        purpose = 3
    elif rowTemp['purpose'] == 'home_improvement':
        purpose = 4
    elif rowTemp['purpose'] == 'house':
        purpose = 5
    elif rowTemp['purpose'] == 'major_purchase':
        purpose = 6 
    elif rowTemp['purpose'] == 'medical':
        purpose = 7 
    elif rowTemp['purpose'] == 'moving':
        purpose = 8 
    elif rowTemp['purpose'] == 'wedding':
        purpose = 9 
    elif rowTemp['purpose'] == 'small_business':
        purpose = 10 
    elif rowTemp['purpose'] == 'vacation':
        purpose = 11
    else:
        purpose = 12       
        
    #print('purpose',purpose) 

    if rowTemp['term'] == ' 36 months':
        term = 1
    if rowTemp['term'] == ' 60 months':
        term = 2
        
    #print('term',term)

    List = List.append(pd.DataFrame({'grade':grade,
                                        'short_emp':rowTemp['short_emp'],
                                        'emp_length_num':rowTemp['emp_length_num'],
                                        'home_ownership':home_ownership,
                                        'dti':rowTemp['dti'],
                                        'purpose':purpose,
                                        'term':term,
                                        'last_delinq_none':rowTemp['last_delinq_none'],
                                        'last_major_derog_none':rowTemp['last_major_derog_none'],
                                        'revol_util':rowTemp['revol_util'],
                                        'total_rec_late_fee':rowTemp['total_rec_late_fee'],
                                        'safe_loans':rowTemp['safe_loans']},index=[i]))
    print(i)
    i = i + 1
    
List.to_csv('loansTemp.csv',encoding='utf-8')    