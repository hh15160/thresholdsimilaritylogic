# Important modules to import
import numpy as np
import pandas as pd
import random
from random import randint
import time
import matplotlib.pyplot as plt
from itertools import product, chain, combinations
import math as math
import seaborn as sns
import scipy as sp



def copula_truth_table(variables, formula, s, beta):
    
    #   Only considers similarity expressions that are invloved in sentence theta
    variables2=[]
    for i in range(0,len(variables)):
        if any(variables[i] in s for s in formula):
            variables2.append(variables[i])
    # Now, varaibles2 contains only relevant s valuess               
    
    # Generates initial 1 and 0s for truth table
    n=len(variables2)
    m = [i for i in product(range(2), repeat=n)] 
    df = pd.DataFrame(m ,columns = variables2)

    column_names=[]
    c=0
    
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
    
    # Further optimisation starts here
    # Remove rows that do not contain a 1
    counts = df[column_name_joint].value_counts()
    drop_value = np.argmax(counts)
    
    for i in range(0,len(df)):
        
        if df[column_name_joint][i]==drop_value:
            df = df.drop([i])

    # Need to add in the s values now
    # Need to find the product of s's 
    s_vals = []
    df = df.reset_index(drop=True)
    
    for ii in range(0,len(df)):
        P=[]
        for jj in range(0, n):
            # If equal to 1, append column name to P
           if df[variables2[jj]][ii]==1:
               P.append(df.columns[jj])
           
        x_sum = copula_atoms(P,variables2,s, beta)
        s_vals.append(x_sum)
        
    df.insert(loc=0, column='S_val', value = s_vals)
    sval_sum = round(np.sum(df['S_val']),4)
    
    if drop_value == 1:
        sval_sum = round(1-sval_sum,4)
        
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


# New formula generator that generates a formula where n defines the number of similarity expressions 
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
            if all(x==connective[0] for x in connective)==False and set(variables).issubset(sub_bracket)==False:
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
    num_variables=n
    s=[]
    variables=[]
    for i in range(0,num_variables):
        s.append(round(random.uniform(0.001, 0.999),4))
        variables.append('s'+str(i+1))
        
    return variables, s

# Function for calculating Franks Copula
def franks_copula(values, beta):
    # Do sum first...
    sum_values = []
    for i in range(0, len(values)):
        p=beta**(values[i])-1
        sum_values.append(p)
    
    prod_sum_values = np.prod(sum_values)
    cop = math.log(1+(prod_sum_values/(beta-1)**(len(values)-1)))/math.log(beta)

    return cop

# Funciton for calculating the copula atoms    
# Will do one atom at a time
def copula_atoms(P,variables,s, beta):
    #Generate F
    F = create_superset(P,variables)
    x_vec = []
    
    # Round values if dependence variable =0 or 1
    if beta==1:
        beta = 0.9999999999999
    if beta==0:
        beta = 0.0000000000001
        
    for i in range(0,len(F)):
        values = f(variables,s,F)
        x = (-1)**(len(F[i])-len(P)) * franks_copula(values[i], beta)
        x_vec.append(x)
        
    x_sum = np.sum(x_vec)
    return x_sum

# Function used for finding supersets
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

# This function works for calculating supersets of P
# Will include empty set too
def create_superset(P,variables): 
    # Initialise variables
    F=[]
    z=[]
    x=[]
    
    for subset in all_subsets(variables):
        z.append(subset)

    for i in range(len(z)):
        x.append(list(z[i]))
        
    # x now contains all subsets of variables    
    for i in range(0,len(x)):
        if set(P).issubset(x[i]):
            F.append(x[i])

    return F

# Function for calculating values for franks copula
# Either mu(si) or 1
def f(variables,s,F):
    values =[]
    for t in range(0,len(F)):
        cop_values = []
        for j in range(0,len(variables)):
            if set([variables[j]]).issubset(F[t]):
                cop_values.append(s[j])
            else:
                cop_values.append(1)
        values.append(cop_values)
    return values    
    

def analysis(n,k, beta):
    # Average time over 100 attempts of n or k
    time2=[]
    for p in range(0,100):
        start_time = time.time()
        formula, variables, s= new_formula_generator(n,k)
        df, sval_sum = copula_truth_table(variables, formula, s, beta)
        end_time = time.time()
        time2.append(end_time-start_time)
        
    time_taken= np.mean(time2)
    std_dev = np.std(time2)
    return df, formula, sval_sum, time_taken, std_dev
  

def run_once(n,k, beta):
        formula, variables, s= new_formula_generator(n,k)
        df, sval_sum = copula_truth_table(variables, formula, s, beta)
        return formula, variables, df, sval_sum
    

def test(beta):
        formula = [['not', 's1', 'and', 's3'],['or'], ['s2']]
        variables = ['s1','s2','s3','s4']
        s = [0.75,0.3,0.12, 0.15]
        df, sval_sum = copula_truth_table(variables, formula, s, beta)
        return formula, variables, df, sval_sum, s
  
  
    
if __name__ == "__main__":    
    #Seaborn for graphs
    sns.set()
    
    
    ###### k=20, n varied ######   
    total_time=[]
    k=5
    n_vec=[]
    err=[]
    beta=0.5
    for n in range(5,1000):
        df, formula, sval_sum, time_taken, std_err = analysis(n,k, beta)
        n_vec.append(n)
        total_time.append(time_taken)
        err.append(std_err)
        print(n)
    plt.xlabel('Number of Similarity Values (n)')
    plt.ylabel('Computational Time (s)')
    plt.scatter(n_vec, total_time, marker = '+') 
    plt.errorbar(n_vec, total_time, err,fmt='none',capsize=5)
    plt.plot(np.unique(n_vec), np.poly1d(np.polyfit(n_vec, total_time, 1))(np.unique(n_vec)), 'r', linewidth=3.0)
    plt.show()   
    
    ######               ######
    
    
    # Fixed number of similarity expressions
    ###### n=1000, k varied ######
    total_time=[]
    n=10
    k_vec=[]
    err=[]
    beta = 0.5
    for k in range(1,20):
        df, formula, sval_sum, time_taken, std_err = analysis(n,k*50, beta)
        k_vec.append(k*50)
        total_time.append(time_taken)
        err.append(std_err)
        print(k)
        
    plt.figure()
    plt.xlabel('Length of Formula (k)')
    plt.ylabel('Computational Time (s)')
    plt.scatter(k_vec, total_time, marker="+", s=100) 
    plt.errorbar(k_vec, total_time, err,fmt='none',capsize=5)
    plt.plot(k_vec, total_time, 'r', linewidth=3)
    #plt.plot(np.unique(k_vec), np.poly1d(np.polyfit(k_vec, total_time, 1))(np.unique(k_vec)), 'r',linewidth=3.0)
    plt.show()
        
    
    # Test a specific formula  
    formula, variables, df, sval_sum, s = test(beta)
    
    
    ####### VARYING DEPENDENCE PLOTS #############
    
    b_vec=[]
    s_vec=[]
    n=3
    k=3
    
    formula, variables, s= new_formula_generator(n,k)
    for beta in range(0,100): 
        beta = beta*0.01
        b_vec.append(beta)
        formula, variables, df, sval_sum, s = test(beta)
        s_vec.append(sval_sum)
        
    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.tight_layout()
    plt.tick_params(labelsize=15)
    plt.title('')
    plt.xlabel(r'$\beta$', fontsize = 'large')
    plt.ylabel(r'$\mu(\theta)$', fontsize = 'large')
    plt.scatter(b_vec, s_vec, marker='+') 

    dependent = s_vec[0]
    independent = s_vec[-1]
    
        
    # Lines for dependence and independence
    plt.plot(np.linspace(-0.5,1.5,200), [dependent]*200, linestyle='--', color = 'C0')
    plt.plot(np.linspace(-0.5,1.5,200), [independent]*200, linestyle='--',color= 'C1')
    
    plt.plot(b_vec, s_vec,'r')
    
    plt.xlim([-0.1,1])
    
    if independent>dependent:
        plt.ylim([dependent-0.01, independent+0.01])
    else:
        plt.ylim([independent-0.01, dependent+0.01])
    
    plt.show()   

