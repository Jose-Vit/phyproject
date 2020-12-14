

import  numpy as np
from  numpy.linalg import solve 
from pylab import meshgrid
import matplotlib.pyplot as plt
from scipy.constants import hbar,m_e
from time import time
import matplotlib.animation as animation


###########################################################
###########################################################
# PROJECT*
# Solving the 2D time-dependent Schrodinger Equation using 
# Crank-Nicolson scheme for a square barrier inside a box 
# with Dirichlet boundary conditions (wave function vanishing)
# at the edges at all times)
# (Includes animation)
# *Adapted from exercise proposed in  University of Toronto
# course PHY407
###########################################################
###########################################################


# Setting constants and parameters (in SI units)


P = 41    # number of x (and y) coordinates not at the edges of square box
L = 1e-9  # size of square box

omega = 1e15
sigma = L/25
kappa1 = 4
kappa2 = 3
m = m_e
sigma = L/25
kappa1 = 4
kappa2 = 10

h = 1e-18 # time step
a = (L/P) # spatial step
N = 300   # number of time steps

x0 = -L/2
xP = L/2
y0 = -L/2
yP = L/2
x_0 = -L/4
y_0 = -L/4



# Defining auxiliary functions

def gaussian(x,y):
    return (np.exp(-(((x-x_0)**2+(y-y_0)**2)/(2*sigma)**2)))*(np.exp(complex(0,kappa1*x+kappa2*y)))

def gaussiansquaredx(x):
    return np.exp(-2*((x-x_0)**2/(2*sigma)**2))
def gaussiansquaredy(y):
    return np.exp(-2*((y-y_0)**2/(2*sigma)**2))

def normconstant(X):            # returns normalization constant
    d = 0                         # psi(x0,t)=psi(xP,t)=0
    for i in range(0,(P-1)**2):
        d+=(np.conj(X[i]))*X[i]
    psi0 = (1/(d*a**2))**(1/2)
    return np.real(psi0)

 
def trapezoidal(x,y):           # trapezoidal rule used to evaluate integrals numerically
    d = 0
    for i in range(0,P-1):
        d+=x[i]*y[i]
    return d*a

start=time()


'''
def potentialfunc(x,y):
    return  ((m*omega**2)/2)*(x**2+y**2)
'''
##########################################################
# Defining matrices to define discretized Hamiltonian as a 
# (P-1)**2x(P-1)**2 matrix
#########################################################


vec_diag1 = np.ones((P-1)*(P-1),complex)
for i in range(len(vec_diag1)):
    vec_diag1[i] = 1
  



B=complex(0,h*hbar/(4*m*a**2))


Sup = B*(np.eye((P-1)**2, k=1,dtype=complex)) 
Sub = B*(np.eye((P-1)**2, k=-1,dtype=complex)) 
UpperSup = B*(np.eye((P-1)**2, k=P-1,dtype=complex)) # band 'above' Sup
LowerSub = B*(np.eye((P-1)**2, k=-(P-1),dtype=complex)) # band 'below' Sub

# Along the bands adjacent to the diagonal,  there are some zero element 
# which can be constructed via 'zeroterm' function. 
def zeroterm(A):
    n=len(A)
    for i in range(n):
        for j in range(n):
            if i%(int(np.sqrt(n)))==0 and j==i-1:
                A[i][j] = 0
                A[j][i] = 0
    return A

zeroterm(Sub)                     
zeroterm(Sup)


H1 =  -(Sub + Sup+ UpperSup + LowerSub) 
H2 =  Sub + Sup+ UpperSup + LowerSub


# Setting the x- and y- points (not including the boundary points)
xpoints = np.zeros(P-1)
ypoints = np.zeros(P-1)
for i in range(0,P-1):
    xpoints[i]= x0+(i+1)*a
    ypoints[i]= y0+(i+1)*a
    

    
# Defining matrix corresponding to square barrier inside the box    
Vmatrix = np.zeros([P-1,P-1])
for i in range(int((P-1)/2)-2,int((P-1)/2)+2):
    Vmatrix[i,int((P-1)/2)-2:int((P-1)/2)+2] = np.ones(len(Vmatrix[i,int((P-1)/2)-2:int((P-1)/2)+2]))
 
Vbarrier = np.ravel(Vmatrix) # transform (P-1)x(P-1) matrix to array of len (P-1)**2
        
  

# Definining the elements along the diagonal
for i in range(len(H1)):
    H1[i][i] = 1+complex(0,((h*hbar)/(2*m*a**2))-(h*Vbarrier[i]/(2*hbar)))
    H2[i][i] = 1+complex(0,((-h*hbar)/(2*m*a**2))-(h*Vbarrier[i]/(2*hbar)))
    


    
  
PSI = [] # list (of length N) to store the psi arrays (of length P-1) at each t 
Normalization = np.zeros(N)
for i in range(N):
    PSI.append([])

# initializing wave function
psi = np.zeros([P-1,P-1],complex)
psisq = np.zeros([P-1,P-1],complex)
for i in range(P-1):
    for j in range(P-1):
        psi[i][j] = gaussian(xpoints[i],ypoints[j])

 
init=np.ravel(psi)    
    

# calculating normalization constant of initial wave via trapezoidal rule    

sx = 0.5*gaussiansquaredx(x0)+0.5*gaussiansquaredx(xP)
sy = 0.5*gaussiansquaredy(y0)+0.5*gaussiansquaredy(yP)
for i in range(0,P-1):
    sx+=gaussiansquaredx(x0+(i+1)*a)
    sy+=gaussiansquaredy(y0+(i+1)*a)
    
I = (a**2)*(sx*sy)
psi0 = (1/I)**(1/2)


PSI[0] = psi0*init     # inserting initial wave function normalized into PSI list
Normalization[0] = psi0   
# loop over time 

##########################################################
# Setting a for loop to get the updated wave function
# (as a vector of len (P-1)**2) by solving a system
# of equations 
#########################################################


for m in range(1,N):
        V = np.dot(H2,PSI[m-1])
        X = solve(H1,V)
        PSI[m] = X*normconstant(X)
    
        
        
        
        
# Constructing the entire wave function vector which includes zero values
# on the boundary conditions

boundarypsi = np.zeros(P+1)
S = np.array([complex(0,0)])

# 'add' inserts zero values where required
def add(X):
    U1 = np.insert(X,P-1,0)
    U2 = np.insert(U1,0,0)
    
    return U2

# auxiliary arrays

PSI_array = np.array(PSI)
wavefns = []
wavefns1 = []
for i in range(len(PSI)):
   wavefns.append([])
   wavefns1.append([])



for i in range(len(PSI)):
    wavefns[i] = np.array(np.reshape(PSI_array[i],(P-1,P-1)))
 

for i in range(len(PSI)):
    j = 0
    while j<P-1:
        wavefns1[i].append(add(wavefns[i][j]))
        j+=1


Wave_array = np.array(wavefns1)

Wave_array1 = np.concatenate((Wave_array,boundarypsi),axis=None)
Wave_array2 = np.concatenate((boundarypsi,Wave_array1),axis=None)

# 'inserting' insert a row of zeros where required corresponding to values at
# edges

def inserting(X):
    A1 = np.concatenate((X,boundarypsi),axis=None)
    A2 = np.concatenate((boundarypsi,A1),axis=None)
    return A2

# 'Wave_functions' stores the  set of complete wave functions(i.e including
# zero values at the boundary points)  
Wave_functions=[]
for i in range(len(PSI)):
    Wave_functions.append(np.ravel(Wave_array[i]))
    Wave_functions[i] = inserting(Wave_functions[i])

    

# Setting the x- and y- points (including the boundary points) useful for 
# animation

x1=np.zeros(P+1)
y1=np.zeros(P+1)
for i in range(P+1):
    x1[i] = x0+i*a
    y1[i] = y0+i*a
    
X1,Y1= meshgrid(x1,y1)   
 

##########################################################
# Animation displaying evolution of wave function
# adapted from outline code provided by stackflow post 
# by 'ImportanceOfBeingErnes'
#########################################################

n0 =  P+1 # Meshsize
fps = 20 # frame per sec
frn = N # frame number of the animation

Z1 = np.zeros((n0, n0, frn))


for i in range(frn):
    Z1[:,:,i] =abs(np.reshape(Wave_functions[i],(P+1,P+1)))


def update_plot(frame_number, Z1, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X1, Y1, Z1[:,:,frame_number], color='deepskyblue')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot = [ax.plot_surface(X1, Y1, Z1[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_zlim(0,round(max(abs(Wave_functions[0]))))
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_zlabel('\u03C8')
ax.set_title('Evolution of wave packet with barrier')
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(Z1, plot), interval=1000/fps)
ani.save('Evolution.mp4')

end=time()

diff=end-start
print(diff)