import random
g=["A","C","G","T"]


def randomizer(g,size):
    j=""
    for _ in range(size):
        z = random.randint(0,3)
        j+=g[z]
        
    return str(j)
        
    
    
    
match = 1
penality = -1
table=[]
s1=randomizer(g,12)
s2=randomizer(g,12)
row = len(s1)
column = len(s2)



def creater(row,column):
    for _ in range(row+1):
        s=[]
        for _ in range (column+1):
            s.append(0)
            
        table.append(s)
   
def filler(arr, current_row, current_col,M,N) :
    
    
    
    if (current_col >= M) :
        return False;
    
    if (current_row >= N) :
        return True;
    
    if(current_row!=0 and current_col!=0):
        table[current_row][current_col] = checker(current_row,current_col)
    
    if (filler(arr, current_row, current_col + 1 ,M,N)):
        return True;
    
    return filler(arr, current_row + 1, 0 ,M , N);


def checker(current_row,current_col):
    
    if(s1[current_row-1]==s2[current_col-1]):
        return table[current_row-1][current_col-1]+match;
    else:
        ret = max([table[current_row-1][current_col-1],table[current_row][current_col-1],table[current_row-1][current_col]])
        fin = ret+penality
        
        if fin>=0:
            return fin
        else:
            return 0



creater(row,column)
filler(table, 0, 0, column+1,row+1)
 
def getalignedseq(x,y,matrix,traceBack):
    xSeq=[]
    ySeq=[]
    i=len(x)
    j=len(y)
    while(i>0 or j>0):
        if traceBack[i][j]=='diag':
            xSeq.append(x[i-1])
            ySeq.append(y[j-1])
            i=i-1
            j=j-1
        elif traceBack[i][j]=='left':
            xSeq.append('-')
            ySeq.append(y[j-1])
            j=j-1
        elif traceBack[i][j]=='up':
            xSeq.append(x[i-1])
            ySeq.append('-')
            i=i-1
        elif traceBack[i][j]=='done':
            break
    return xSeq,ySeq
            