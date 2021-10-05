#Standard imports
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import tinyarray
import time
import sys
import os

sigma_0 = tau_0 = nu_0 = tinyarray.array([[1,0],[0,1]])
sigma_x = tau_x = nu_x = tinyarray.array([[0,1],[1,0]])
sigma_y = tau_y = nu_y = tinyarray.array([[0,-1j],[1j,0]])
sigma_z = tau_z = nu_z = tinyarray.array([[1,0],[0,-1]])
sigma_up=tinyarray.array([[1,0],[0,0]])
sigma_dw=tinyarray.array([[0,0],[0,1]])


#FUNCTIONS
#_________________________________________________________________________________________

def MatrixStructure(L):
    """A function that creates all the connections between sites for a square system with length and width L."""
    PtoV=np.zeros((L**2,4))
    for k in range(L**2):
        x=k%L
        y=k//L
        PtoV[k,0]=x+y*L
        PtoV[k,1]=((x+1)%L)+y*L
        PtoV[k,2]=x+((y+1)%L)*L
        PtoV[k,3]=((x+1)%L)+((y+1)%L)*L     
    return(PtoV) 

def TB_Operators(L):
    """A function that creates all the tight binding operators for a square system with length and width L."""
    PtoV=MatrixStructure(L)
    phi=np.zeros((L**2,L**2))
    phiX=np.zeros((L**2,L**2))
    phiY=np.zeros((L**2,L**2))
    Dx=np.zeros((L**2,L**2))
    Dy=np.zeros((L**2,L**2))
    for i in range(L**2):
        x=i%L
        y=i//L
        phi[i,int(PtoV[i,0])]=1/4
        phi[i,int(PtoV[i,1])]=1/4*(-1)**((x//(L-1))*aperiodic)
        phi[i,int(PtoV[i,2])]=1/4*(-1)**((y//(L-1))*aperiodic)
        phi[i,int(PtoV[i,3])]=1/4*(-1)**((x//(L-1))*aperiodic)*(-1)**((y//(L-1))*aperiodic)
        Dx[i,int(PtoV[i,0])]=-1/2
        Dx[i,int(PtoV[i,1])]=1/2*(-1)**((x//(L-1))*aperiodic)
        Dx[i,int(PtoV[i,2])]=-1/2*(-1)**((y//(L-1))*aperiodic)
        Dx[i,int(PtoV[i,3])]=1/2*(-1)**((x//(L-1))*aperiodic)*(-1)**((y//(L-1))*aperiodic)
        Dy[i,int(PtoV[i,0])]=-1/2
        Dy[i,int(PtoV[i,1])]=-1/2*(-1)**((x//(L-1))*aperiodic)
        Dy[i,int(PtoV[i,2])]=1/2*(-1)**((y//(L-1))*aperiodic)
        Dy[i,int(PtoV[i,3])]=1/2*(-1)**((x//(L-1))*aperiodic)*(-1)**((y//(L-1))*aperiodic)
        phiX[i,int(PtoV[i,0])]=1/2
        phiX[i,int(PtoV[i,1])]=1/2*(-1)**((x//(L-1))*aperiodic)
        phiY[i,int(PtoV[i,0])]=1/2
        phiY[i,int(PtoV[i,2])]=1/2*(-1)**((y//(L-1))*aperiodic) 
    return phi,phiX,phiY,Dx,Dy 



#SCRIPT
#_________________________________________________________________________________________

start_time = time.time() 

#INPUT PARAMETER -> Parameter for disorder iteration when running on the cluster
it = int(sys.argv[1])


#PARAMETERS
Ndata=20 #Amount of Disorder iterations calculated on a single cluster job
Disorder=30 #DisorderStrength
L=101 #Alsways odd
Neig=20 #Number of calculated Eigenstates
#Selecting the desired symmetry class of the system
Type="D"
#Type="AIII"
#Type="AII"
#Type="A"
aperiodic=1 
L=L-aperiodic #To make sure that the system is Even for anti-PBC and Odd for PBC
Center=0 #Energy around which we are calculating eigenvalues
Sigma =0 # Possible smoothening factor, not used in this paper

#DATA ARRAYS
DataEig=np.zeros((Ndata,Neig))

start_time=time.time()
print("Time_Start: ", time.time()-start_time)


#Defining non-symmetrized Stacey tight-binding operators from equations 2.1.
phi,phiX,phiY,Dx,Dy=TB_Operators(L)
print("Time_MatrixCreation: ", time.time()-start_time)

#Creating Sparse matrices required for the Hamiltonian   
D_sigma_x=sc.sparse.kron(sc.sparse.csr_matrix(sigma_x),sc.sparse.csr_matrix(Dx),format="csr")
D_sigma_y=sc.sparse.kron(sc.sparse.csr_matrix(sigma_y),sc.sparse.csr_matrix(Dy),format="csr")
print("Time_D: ", time.time()-start_time) 
Phi=sc.sparse.kron(sc.sparse.csr_matrix(sigma_0),sc.sparse.csr_matrix(phi),format="csr")
PhiD=sc.sparse.kron(sc.sparse.csr_matrix(sigma_0),sc.sparse.csr_matrix(phi).transpose(),format="csr")
print("Time_Phi: ", time.time()-start_time) 

#Defining additional operators PhiX and PhiY for the AIII class of system
if Type == "AIII":
    PhiX=sc.sparse.kron(sc.sparse.csr_matrix(sigma_0),sc.sparse.csr_matrix(phiX),format="csr")
    PhiXD=sc.sparse.kron(sc.sparse.csr_matrix(sigma_0),sc.sparse.csr_matrix(phiX).transpose(),format="csr")
    print("Time_PhiX: ", time.time()-start_time) 
    PhiY=sc.sparse.kron(sc.sparse.csr_matrix(sigma_0),sc.sparse.csr_matrix(phiY),format="csr")
    PhiYD=sc.sparse.kron(sc.sparse.csr_matrix(sigma_0),sc.sparse.csr_matrix(phiY).transpose(),format="csr")
    print("Time_PhiY: ", time.time()-start_time)
print("Time_Matrices: ", time.time()-start_time)

#CREATING APPROPRIATE RANDOM ARRAYS 
for i in range(Ndata):
    start_time=time.time()
    np.random.seed(i+Ndata*it)
    Values1=np.random.rand(L**2)*Disorder
    Values2=np.random.rand(L**2)*Disorder
    ValGrid1=Values1.reshape(L,L)
    ValGrid2=Values2.reshape(L,L)
    #Adding possible smoothening not used in this paper
    ValGrid1=sc.ndimage.gaussian_filter(ValGrid1, sigma=Sigma)
    ValGrid2=sc.ndimage.gaussian_filter(ValGrid2, sigma=Sigma)
    Values1=(ValGrid1.flatten()/np.max(ValGrid1))*Disorder
    Values2=(ValGrid2.flatten()/np.max(ValGrid2))*Disorder
    Average1=np.average(Values1)
    Average2=np.average(Values2)
    Rand1=sc.sparse.diags(Values1-Average1,format="csr")
    Rand2=sc.sparse.diags(Values2-Average2,format="csr")
#CREATING THE APPROPRIATE HAMILTONIAN
    U=-1j*D_sigma_x-1j*D_sigma_y
    #U is the non symmetrized Stacey-hamiltonian from equation 2.3 in position space
    P=PhiD@Phi 
    print("Time_Start: ", time.time()-start_time)
    #The next part symmetrizes the Hamiltonian to create the operators H and P needed for the generalized eigenvalue problem in equation 5.1
    if Type== "AIII":
        Ax=sc.sparse.kron(sc.sparse.csr_matrix(sigma_x),Rand1/np.sqrt(2),format="csr")
        Ay=sc.sparse.kron(sc.sparse.csr_matrix(sigma_y),Rand2/np.sqrt(2),format="csr")
        H=PhiD@U+PhiYD@Ax@PhiY+PhiXD@Ay@PhiX
    if Type== "AII":
        V=sc.sparse.kron(sc.sparse.csr_matrix(sigma_0),Rand1,format="csr")
        H=PhiD@U+PhiD@V@Phi
    if Type== "A":
        V=sc.sparse.kron(sc.sparse.csr_matrix(sigma_0),Rand1/np.sqrt(2),format="csr")
        Mz=sc.sparse.kron(sc.sparse.csr_matrix(sigma_z),Rand2/np.sqrt(2),format="csr")
        H=PhiD@U+PhiD@V@Phi+PhiD@Mz@Phi
    if Type== "D":
        Mz=sc.sparse.kron(sc.sparse.csr_matrix(sigma_z),Rand1,format="csr")
        H=PhiD@U+PhiD@Mz@Phi
    print("Time_Matrix: ", time.time()-start_time)

#DIAGONALIZATION
    Eigs=sc.sparse.linalg.eigsh(H,M=P,k=Neig,sigma=Center,which="LM",return_eigenvectors=False,tol=1e-6)
    EigsSort=np.sort(Eigs)
    DataEig[i]=EigsSort
    print(i)
    print("Time_Total: ", time.time()-start_time)
    if i==0:
        print("Estimated_time: ", (time.time()-start_time)*Ndata)
    
print(DataEig)

#DEFINE A FOLDER FOR SAVING
#_________________________________________________________________________________________

#np.save("/DataSaveDirectory/" +"DOS_DataSpTest_"+str(Type)+"_AP_"+str(int(aperiodic))+"_it_"+str(int(it*Ndata))+"_"+str(int(it*Ndata+Ndata))+"_L_"+str(L)+"_Dis_"+str(int(Disorder*10))+"_SG_"+str(int(Center)),DataEig)



