# Important modules to import
import numpy as np
import pandas as pd
import random
from random import randint
import time
import matplotlib.pyplot as plt
import seaborn as sns


# Orders the similarty expressions from smallest to largest
def sort_expressions(s):
    s = np.array(s)
    order = np.argsort(s)
    #Sorts s into order smallest to largest
    s.sort(axis=0)
    n=np.size(s)
    mu=np.array(np.zeros(n+1))
    mu[0]=s[0]
    for i in range(1, n):
        mu[i]=s[i]-s[i-1]
        #print(mu[i])
    mu[n]=1-s[-1]  
    return [s, mu, order]

# Creates the truth table
def create_truth_table(variables, formula, s):
    variables3=[]
    s_new = []
    
#   Only considers similarity expressions that are invloved in sentence theta
    for i in range(0,len(variables)):
        if any(variables[i] in s for s in formula):
            variables3.append(variables[i])
            s_new.append(s[i])
    [s, mu, order] = sort_expressions(s_new)
    n=len(variables3)
    variables2=[]
    order = order[::-1]
    for i in range(0,n):
        variables2.append(variables3[order[i]])
    m = np.ones((n+1,n))
    m=np.tril(m, -1)
    rev_mu = mu[::-1]
    m=np.asarray(m)
    df = pd.DataFrame(m,columns= variables2)
    df.insert(loc=0, column='S_val', value = rev_mu)
    
    column_names=[]
    c=0
    
#   Loops through the entire formula
    for i in range(0,len(formula)):
        # Checks in a sub bracket
        if formula[i][0]!='or':
            #Creates column name for header of data frame
            seperator = ' '
            column_names.append(seperator.join(formula[i]))
            sval=[]
            zero=False
            #Looking through individual formula
            for j in range(0, len(formula[i])):
                if formula[i][j]=='not':
                    zero=True
                    # Goes onto next j value if zero is true
                    continue
                elif formula[i][j]=='and':
                    continue
                else:
                    if zero==True:
                        sval.append([formula[i][j],0])
                        zero=False
                    else:
                        sval.append([formula[i][j],1])
                        
            if len(formula[i])==1:
                truth =np.array(np.zeros(len(df)))
                #When formula length =1 (ie when only one value in subbracket), no need to calculate values
                for k in range(0,len(df)):
                    truth[k]=df[sval[0][0]][k]
                    
            # Number in sub bracket is greater than 1
            else:
                truth =np.array(np.zeros(len(df)))
                for l in range(0,len(df)):
                    istrue = True
                    for k in range(0,len(sval)):
                        if df[sval[k][0]][l]==sval[k][1]:
                            istrue=True
                        else:
                            istrue=False
                            break
                    if istrue==True:
                        truth[l]=1
                    else:
                        truth[l]=0
                        
            df[column_names[c]]=truth
            c=c+1;
        
    # Now bring all togther by evaluating all rows - all joined by an 'or'
    truth =np.array(np.zeros(len(df)))
    seperator = ') or ('
    column_name_joint = '(' + seperator.join(column_names) + ')'
    for i in range(0,len(df)):
        istrue = True
        for j in range(0,len(column_names)):
            if df[column_names[j]][i]==1:
                istrue=True
                break
            else:
                istrue=False
        if istrue==True:
            truth[i]=1
        else:
            truth[i]=0
            
    df[column_name_joint]=truth
    sval_sum = evaluation(df,column_name_joint)
    return df, sval_sum
    

#Evaluation function   
def evaluation(df,column_name_joint):
    #Need to find the sum of svals
    sval_sum=0
    for i in range(0,len(df)):
        if df[column_name_joint][i]==1:
            sval_sum=sval_sum+df['S_val'][i]
    sval_sum = round(sval_sum,4)
    return sval_sum


# Formula generator that generates a formula where n defines the number of similarity expressions 
# k defines the length of the formula
# or's are between sub_brackets, and's are in sub_brackets
def new_formula_generator(n,k):
    [variables, s]= create_variables(n)
    connective_list = ['and', 'or']
    variable_list = variables
    formula=[]
    connective=[]
    
    # Number of connectives is k-1
    for j in range(0,k-1):
        connective.append(random.choice(connective_list))
    
    #Create an empty sub brakcket
    sub_bracket=[]
    
    for i in range(0,k):   
        # Decide on whether a 'not' is present or not
        if randint(0,1)==1:
            not_present=True
        else:
            not_present=False    
        s_value = random.choice(variable_list)
        
        # Keep iterating through s_value if in formula subbracket already
        while (s_value in sub_bracket)==True:
            # Stops loop from getting stuck if k>n
            if all(x==connective[0] for x in connective)==False:
                s_value = random.choice(variable_list)
            else:
                break
             
#       Checks for 'not's
        if not_present==True:
            #if first loop through j
            sub_bracket.append('not')
        
        sub_bracket.append(s_value)
        
        if i!=(k-1):
            
            if connective[i]=='and':
                sub_bracket.append(connective[i])
                
            elif connective[i]=='or':
                # Close bracket and add to formula
                formula.append(sub_bracket)
                # Add connective on its own
                formula.append([connective[i]])
                sub_bracket=[]
                
        else:
            formula.append(sub_bracket)
             
    return [formula, variables, s]
            


def create_variables(n):
#   Initialise variables and vectors
    num_variables=n
    s=[]
    variables=[]
    
    for i in range(0,num_variables):
        s.append(round(random.uniform(0.001, 0.999),4))
        variables.append('s'+str(i+1))
        
    return variables, s



def analysis(n,k):
    # Average time over 100 attempts of n
    time2=[]
    
    for p in range(0,100):
        start_time = time.time()
        formula, variables, s= new_formula_generator(n,k)
        df, sval_sum = create_truth_table(variables, formula, s)
        end_time = time.time()
        time2.append(end_time-start_time)
        
    time_taken= np.mean(time2)
    return df, formula, sval_sum, time_taken, s
 
    
# Code to run analysis once
def run_once(n,k):
        formula, variables, s= new_formula_generator(n,k)
        df, sval_sum = create_truth_table(variables, formula, s)
        return formula, variables, df, sval_sum
  
    
# Code for testing a specific formula
def test():
        formula = [['not','s1','and','not','s2','and','not','s3']]
        variables = ['s1','s2','s3','s4','s5']
        s = [0.75,0.3,0.12, 0.33, 0.98]
        df, sval_sum = create_truth_table(variables, formula, s)
        return formula, variables, df, sval_sum, s    



if __name__ == "__main__":    
    #Seaborn for graphs
    sns.set()
    total_time=[]
    
    
    ###### k=15, n varied ######
    k=15
    n_vec=[]
    for n in range(1,100):
        df, formula, sval_sum, time_taken, s = analysis((n*10)+5,k)
        n_vec.append((n*10)+5)
        total_time.append(time_taken)
        print(n)
        
    plt.xlabel('Number of similarity values (n)')
    plt.ylabel('Computational Time (s)')
    plt.scatter(n_vec, total_time,marker="+") 
    plt.plot(np.unique(n_vec), np.poly1d(np.polyfit(n_vec, total_time, 2))(np.unique(n_vec)), 'r', linewidth=3.0)
    plt.show()   
    
    ######               ######
    
    # Test a specific formula  
    #formula, variables, df, sval_sum, s = test()
    
    
    # Fixed number of similarity expressions
    ###### n=1000, k varied ######
    n=1000
    k_vec=[]
    for k in range(1,100):
        df, formula, sval_sum, time_taken, s = analysis(n,k*10)
        k_vec.append(k*10)
        total_time.append(time_taken)
        print(k)
    plt.figure()
    plt.xlabel('Length of Formula (k)')
    plt.ylabel('Computational Time (s)')
    plt.scatter(k_vec, total_time,marker="+") 
    plt.plot(np.unique(k_vec), np.poly1d(np.polyfit(k_vec, total_time, 2))(np.unique(k_vec)), 'r',linewidth=3.0)
    plt.show()
    
    ######              ######
        
        
    
    
    
    