import GPyOpt
import numpy as np
#import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

in_file='input_mana05737.txt'
phi_ng=np.loadtxt(in_file)[:,0]+1j*np.loadtxt(in_file)[:,1]

NG_x = np.linspace(-20, 20, num=400, endpoint=True)
y = phi_ng
NG_phi = interp1d(NG_x, y, kind='linear')


def target(q0,disp_target):      
    target_xi=-np.log(10**(0.25))
    target_gate_strength=0.047*2*np.sqrt(2)
    exp_coeff=-np.exp(2*target_xi)*q0**2
    phi_target=(np.pi/2)**(-0.25)*np.exp(target_xi/2)*np.exp(exp_coeff)*np.exp(1j*target_gate_strength*q0**3)*np.exp(-1j*q0*disp_target)
    return phi_target

def NG(q0):
    if abs(q0)>10:
        NG_v=0
    else: NG_v=NG_phi(q0)
    return NG_v

def squeezed_wavefunction_complex(xi, position, q3, p3):
    xi_abs=np.abs(xi)
    xi_f=xi*(np.tanh(xi_abs))/xi_abs
    xi_f_abs=np.abs(xi_f)
    xi_c=(1-xi_f_abs**2)**(0.25)/np.sqrt(1-xi_f)
    phisq_xx = (-1)*((position - q3)**2)*(1+xi_f)/(1-xi_f) 
    phisq_x =  2j * p3 * position 
    phisq_x0 =  -1j*p3*q3
    phi_sq = (np.pi/2)**(-0.25) *xi_c* np.exp(phisq_xx+phisq_x+phisq_x0) 
    return phi_sq

def squeezed_wavefunction(xi, position, q3, p3):
    """
    xi is real, no angle here
    """
    xi_abs=np.abs(xi)
    
    phisq_xx = (-1)*((position - q3)**2)*np.exp(2*xi_abs)
    phisq_x =  2j * p3 * position 
    phisq_x0 =  -1j*p3*q3
    phi_sq = (np.pi/2)**(-0.25) * np.exp(phisq_xx+phisq_x+phisq_x0)*np.exp(xi_abs/2)
    return phi_sq    
def probability_integrand(q, q2,theta, xi, q3, p3):
   
    position_value = q*np.cos(theta) + q2*np.sin(theta)
    psi_input = NG(position_value)
    mode2_position = -q*np.sin(theta) + q2*np.cos(theta)
    phi_sq = squeezed_wavefunction(xi, mode2_position, q3, p3)
    probability_val = (abs(psi_input) **2)*(abs(phi_sq)**2)
    return probability_val

def prob(qn,eta,num_q,num_q2,bound_q2,theta, xi, q3, p3):

    q2_arr=np.linspace(-bound_q2,bound_q2,num_q2)
    q_arr=np.linspace(qn-eta,qn+eta,num_q)
    prob_list=[] 
    for j in q_arr:
        q = j
        n_q = 1.0  # All elements except the ones at the boundaries have to be multiplied by 1.0.                        
        if(j == q_arr[0] or j == q_arr[len(q_arr)-1]):  # if j is the first or last element of the array, then n_q = 0.5
            n_q = 0.5
        for i in q2_arr:
            q2 = i
            n_q2 = 1.0
            if(i == q2_arr[0] or i == q2_arr[len(q2_arr)-1]):  # if j is the first or last element of the array, then n_q = 0.5
                n_q2 = 0.5
            prob_list.append(n_q*n_q2*probability_integrand(q, q2,theta, xi, q3, p3))            
    prob_sum=sum(prob_list)
    prob_val=prob_sum*2*bound_q2/(num_q2-1)*2*eta/(num_q-1)
    return prob_val


def overlap_integrand(q, q2, q0,
             theta, r, xi, q3, p3,disp_target):        
    position_value = q*np.cos(theta) + q2*np.sin(theta)
    psi_input =NG(position_value)
    mode2_position = -q*np.sin(theta) + q2*np.cos(theta)
    phi_sq = squeezed_wavefunction(xi, mode2_position, q3, p3)
    a=-2*q2*q0/np.sin(r)+(q2**2+q0**2)*np.cos(r)/np.sin(r)
    f_integral =np.sqrt(2/np.pi)/np.sqrt(1-np.exp(-2j*r))*(np.exp(1j*a))
    phi_target_conj=np.conjugate(target(q0,disp_target))
    overlap_integrand=psi_input*phi_sq *f_integral*phi_target_conj
    return overlap_integrand

overlap_integrand_q0q2 = lambda q,q2, q0: (overlap_integrand(q,q2, q0,theta, r, xi, q3, p3,disp_target))

def overlap_q0q2(q_val,bound_q0,bound_q2,num_q0,num_q2,theta, r, xi, q3, p3,disp_target):
    q=q_val
    q0_arr=np.linspace(-bound_q0,bound_q0,num_q0)
    q2_arr=np.linspace(-bound_q2,bound_q2,num_q2)
    overlap_q0q2_list=[]
    for j in q0_arr:
        q0=j
        n_q0 = 1.0  # All elements except the ones at the boundaries have to be multiplied by 1.0.                        
        if(j == q0_arr[0] or j == q0_arr[len(q0_arr)-1]): 
            n_q0 = 0.5
        for i in q2_arr:
            q2=i
            n_q2 = 1.0
            if(i == q2_arr[0] or i == q2_arr[len(q2_arr)-1]):  # if j is the first or last element of the array, then n_q = 0.5
                n_q2 = 0.5
      

            overlap_q0q2_list.append(n_q2*n_q0*overlap_integrand(q, q2, q0,theta, r, xi, q3, p3,disp_target))                
    overlap_integrand_val=sum(overlap_q0q2_list)*2*bound_q0*2*bound_q2/(num_q0-1)/(num_q2-1)
    overlap_integrand_val2=abs(overlap_integrand_val)**2
    return overlap_integrand_val2

def overlap_eta(qn,eta,num_q,num_q0,num_q2,bound_q0,bound_q2,theta, r, xi, q3, p3,disp_target):
    overlap_list=[]
    q_arr=np.linspace(qn-eta,qn+eta,num_q)
    for j in q_arr:
        q = j
        n_q = 1.0  # All elements except the ones at the boundaries have to be multiplied by 1.0.                        
        if(j == q_arr[0] or j == q_arr[len(q_arr)-1]):  # if j is the first or last element of the array, then n_q = 0.5
            n_q = 0.5
        
        overlap_list.append(n_q*overlap_q0q2(q,bound_q0,bound_q2,num_q0,num_q2,theta, r, xi, q3, p3,disp_target))
    overlap_val=np.sum(overlap_list)*2*eta/(num_q-1)
    return overlap_val 

def fid_val(eta,qn,num_q,num_q0,num_q2,bound_q0,bound_q2,theta, r, xi, q3, p3,disp_target):    
    over=overlap_eta(qn,eta,num_q,num_q0,num_q2,bound_q0,bound_q2,theta, r, xi, q3, p3,disp_target)
    pro=prob(qn,eta,num_q,num_q2,bound_q2,theta,xi, q3, p3)
    fid=over/pro
    return fid

fidelity_f = lambda theta,r, xi,q3,p3,disp_target: (fid_val(eta,qn,num_q,num_q0,num_q2,bound_q0,bound_q2,theta, r, xi, q3, p3,disp_target))



def outputq_q0(q,q0,num_q2,bound_q2):
    q2_list=np.linspace(-bound_q2,bound_q2,num_q2)
    outputqq0_q2_list=[]
    for i in q2_list:
        outputqq0_q2_list.append(output_qq2q0(q,q0,i))
    outputqq0_val=sum(outputqq0_q2_list)*2*bound_q2/num_q2
    return outputqq0_val



def out():

    q0_list=np.linspace(-bound_q0,bound_q0,num_q0)
    q_list=np.linspace(-bound_q,bound_q,num_q)
    for j in q_list:
        outputq_q0_list=[]
        q=j
        for i in q0_list:
            outputq_q0_list.append(((outputq_q0(q,i,num_q2,bound_q2))))
    N=0
    dx=2*bound_q0/num_q0 
    for i in outputq_q0_list:
        N=i**2*dx+N
    RN=abs(N) 
    FN=RN/np.math.sqrt(N)
    R_outputq_q0_list=[]
    for i in outputq_q0_list:
        R_outputq_q0_list.append(i/FN)
    return q0_list,R_outputq_q0_list

def fidelity_a(x):
#    x=GPyOpt.util.general.reshape(x, 4)
    theta=x[:,0]
    r=-np.pi/2
    q3=x[:,1]
    xi=x[:,2]
    p3=0
    disp_target=x[:,3]
    f_val=1-fidelity_f(theta,r, xi,q3,p3,disp_target)
    return np.array([f_val])



''' The global minimum in dimension d is at x1=x2=...=xd=420.9687 '''


domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.1, 1.5)},
          {'name': 'var_2', 'type': 'continuous', 'domain': (0, 1)},
          {'name': 'var_3', 'type': 'continuous', 'domain': (0.1, 1)},
          {'name': 'var_4', 'type': 'continuous', 'domain': (-3,0)}]

#X_init = np.array([[0., 0., 0., 0., 0., 0.]])
#Y_init = schwefel(X_ini)
eta=0.1
qn=0

num_qq0q2=50
num_q=num_qq0q2
num_q0=num_qq0q2
num_q2=num_qq0q2
bound_q0=7
bound_q2=5
#X_init =np.array([[2.43115296, 0.1, 0., 0.,-0.08,2.64661375]])
X_init=np.array([[ 1,        .7 , 0.4   ,      -1]])
Y_init = fidelity_a(X_init)
#Y_init=0.5


iter_count =1000
current_iter = 0
X_step = X_init
Y_step = Y_init
print ('X_init',X_init)
print ('Y_init',1-Y_init)



import time
start = time. time()
     
best_fid=0.8
best_para=[] 
while current_iter < iter_count:
    bo_step = GPyOpt.methods.BayesianOptimization(f = None, domain = domain, X = X_step, Y = Y_step)
    x_next = bo_step.suggest_next_locations()
    y_next = fidelity_a(x_next)
    X_step = np.vstack((X_step, x_next))
    Y_step = np.vstack((Y_step, y_next))
    print 'iter=',current_iter
    if 1-y_next>best_fid:
        best_fid=1-y_next
        best_para=x_next
        print 'iter=',current_iter
        print 'best_fid=',best_fid
        print 'best_para',best_para  
        print '====================='
    current_iter += 1
#Y_step_list=[]
# Print results of optimization
#print 'x:',X_step
#for i in range(iter_count):
#    Y_step_list.append(1-Y_step[i])
  #  print 
#    print 'theta,r,q3,p3,xi_re,xi_im:',X_step[i]
 #   print 'f(x):',1-Y_step[i]

#print 'best_fid=',best_fid
#print 'best_para',best_para
#iter_count_list=np.linspace(1,iter_count,iter_count)  


end = time. time()
print("the running time=",end - start)


