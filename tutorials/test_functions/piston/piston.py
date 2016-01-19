##########################################################################
#
# PISTON SIMULATION FUNCTION
#
# Authors: Sonja Surjanovic, Simon Fraser University
#          Derek Bingham, Simon Fraser University
# Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
#
# Copyright 2013. Derek Bingham, Simon Fraser University.
#
# THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
# FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
# derivative works, such modified software should be clearly marked.
# Additionally, this program is free software; you can redistribute it 
# and/or modify it under the terms of the GNU General Public License as 
# published by the Free Software Foundation; version 2.0 of the License. 
# Accordingly, this program is distributed in the hope that it will be 
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# For function details and reference information, see:
# http://www.sfu.ca/~ssurjano/
#
##########################################################################
#
# OUTPUT AND INPUT:
#
# C = cycle time
# xx = [M, S, V0, k, P0, Ta, T0]
#
#########################################################################
#M = [30,60];
#S = [0.005,0.020];
#v0 =  [0.002,0.010];
#k = [1000, 5000];
#p0 = [90000, 111000];
#Ta = [290,296];
#T0 = [340,360];
import numpy as np      
def fun(xx):
    pi = np.pi
    # Scaling input from [-1,1] hypercube to the input parameter ranges
    M  = (xx[0]+1)*0.5*(60-30)+30;
    S  = (xx[1]+1)*0.5*(0.02-0.005)+0.005;
    V0 = (xx[2]+1)*0.5*(0.010-0.002)+0.002;
    k  = (xx[3]+1)*0.5*(5000-1000)+1000;
    P0 = (xx[4]+1)*0.5*(110000-90000)+90000;
    Ta = (xx[5]+1)*0.5*(296-290)+290;
    T0 = (xx[6]+1)*0.5*(360-340)+340;
    
    Aterm1 = P0 * S;
    Aterm2 = 19.62 * M;
    Aterm3 = -k*V0 / S;
    A = Aterm1 + Aterm2 + Aterm3;
    
    Vfact1 = S / (2*k);
    Vfact2 = np.sqrt(A**2 + 4*k*(P0*V0/T0)*Ta);
    V = Vfact1 * (Vfact2 - A);
    
    fact1 = M;
    fact2 = k + (S**2)*(P0*V0/T0)*(Ta/(V**2));
    
    C = 2 * pi * np.sqrt(fact1/fact2);

    out = np.array([0.314159E1*(M*(k+4*k**2*P0*T0**(-1)*Ta*V0*((-0.1962E2)*M+  \
                    (-1)*P0*S+k*S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*  \
                    M+P0*S+(-1)*k*S**(-1)*V0)**2)**0.5E0)**(-2))**(-1))**(-0.5E0)  \
                    *(8*k**2*M*P0*T0**(-1)*Ta*V0*((-0.1962E2)+0.1962E2*(  \
                    0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)*(4*k*P0*T0**(-1)*Ta*  \
                    V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)**2)**(-0.5E0))*((  \
                    -0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*  \
                    V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)**2)**0.5E0)**(-3)*(k+  \
                    4*k**2*P0*T0**(-1)*Ta*V0*((-0.1962E2)*M+(-1)*P0*S+k*S**(  \
                    -1)*V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*  \
                    S**(-1)*V0)**2)**0.5E0)**(-2))**(-2)+(k+4*k**2*P0*T0**(-1)*  \
                    Ta*V0*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(4*k*P0*  \
                    T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)**2)  \
                    **0.5E0)**(-2))**(-1)),
                    0.251327E2*k**2*M*P0*T0**(-1)*Ta*V0*((-1)*P0+(-1)*k*S**(  \
                    -2)*V0+0.1E1*(P0+k*S**(-2)*V0)*(0.1962E2*M+P0*S+(-1)*k*  \
                    S**(-1)*V0)*(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)  \
                    *k*S**(-1)*V0)**2)**(-0.5E0))*((-0.1962E2)*M+(-1)*P0*S+k*  \
                    S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*  \
                    k*S**(-1)*V0)**2)**0.5E0)**(-3)*(M*(k+4*k**2*P0*T0**(-1)*  \
                    Ta*V0*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(4*k*P0*  \
                    T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)**2)  \
                    **0.5E0)**(-2))**(-1))**(-0.5E0)*(k+4*k**2*P0*T0**(-1)*Ta*  \
                    V0*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(4*k*P0*T0**(-1)  \
                    *Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)**2)**0.5E0)**(  \
                    -2))**(-2),
                   (-0.125664E2)*k*P0*Ta*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*  \
                    V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)  \
                    *V0)**2)**0.5E0)**(-3)*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*  \
                    V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)  \
                    *V0)**2)**0.5E0+(-2)*k*S**(-2)*V0*(S+T0**(-1)*((-0.1962E2)*  \
                    M*S*T0+P0*S**2*((-0.1E1)*T0+0.2E1*Ta)+0.1E1*k*T0*V0)*(  \
                    4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*  \
                    V0)**2)**(-0.5E0)))*(T0+4*k*P0*Ta*V0*((-0.1962E2)*M+(-1)*  \
                    P0*S+k*S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*  \
                    S+(-1)*k*S**(-1)*V0)**2)**0.5E0)**(-2))**(-1)*(M*(k+4*k**2*  \
                    P0*T0**(-1)*Ta*V0*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(  \
                    4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*  \
                    V0)**2)**0.5E0)**(-2))**(-1))**0.5E0,
                   (-0.314159E1)*M**(-1)*(1+(-8)*k**2*P0*S**(-2)*T0**(-1)*Ta*  \
                    V0**2*(S+T0**(-1)*((-0.1962E2)*M*S*T0+P0*S**2*((-0.1E1)*  \
                    T0+0.2E1*Ta)+0.1E1*k*T0*V0)*(4*k*P0*T0**(-1)*Ta*V0+(  \
                    0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)**2)**(-0.5E0))*((  \
                    -0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*  \
                    V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)**2)**0.5E0)**(-3)+8*  \
                    k*P0*T0**(-1)*Ta*V0*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*  \
                    V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)  \
                    *V0)**2)**0.5E0)**(-2))*(M*(k+4*k**2*P0*T0**(-1)*Ta*V0*((  \
                    -0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*  \
                    V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)**2)**0.5E0)**(-2))**(  \
                    -1))**0.15E1,
                   (-0.314159E1)*M*((-8)*k**2*P0*T0**(-1)*Ta*V0*((-1)*S+  \
                    0.5E0*(4*k*T0**(-1)*Ta*V0+2*S*(0.1962E2*M+P0*S+(-1)*k*  \
                    S**(-1)*V0))*(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)  \
                    *k*S**(-1)*V0)**2)**(-0.5E0))*((-0.1962E2)*M+(-1)*P0*S+k*  \
                    S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*  \
                    k*S**(-1)*V0)**2)**0.5E0)**(-3)+4*k**2*T0**(-1)*Ta*V0*((  \
                    -0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*  \
                    V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)**2)**0.5E0)**(-2))*(  \
                    M*(k+4*k**2*P0*T0**(-1)*Ta*V0*((-0.1962E2)*M+(-1)*P0*S+  \
                    k*S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)  \
                    *k*S**(-1)*V0)**2)**0.5E0)**(-2))**(-1))**(-0.5E0)*(k+4*  \
                    k**2*P0*T0**(-1)*Ta*V0*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)  \
                    *V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(  \
                    -1)*V0)**2)**0.5E0)**(-2))**(-2),
                   (-0.314159E1)*M*((-0.16E2)*k**3*P0**2*T0**(-2)*Ta*V0**2*(  \
                    4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*  \
                    V0)**2)**(-0.5E0)*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(4*  \
                    k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)  \
                    **2)**0.5E0)**(-3)+4*k**2*P0*T0**(-1)*V0*((-0.1962E2)*M+(-1)  \
                    *P0*S+k*S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+  \
                    P0*S+(-1)*k*S**(-1)*V0)**2)**0.5E0)**(-2))*(M*(k+4*k**2*  \
                    P0*T0**(-1)*Ta*V0*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(  \
                    4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*  \
                    V0)**2)**0.5E0)**(-2))**(-1))**(-0.5E0)*(k+4*k**2*P0*T0**(-1)  \
                    *Ta*V0*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(4*k*P0*  \
                    T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)**2)  \
                    **0.5E0)**(-2))**(-2),
                   (-0.314159E1)*M*(0.16E2*k**3*P0**2*T0**(-3)*Ta**2*V0**2*(  \
                    4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*  \
                    V0)**2)**(-0.5E0)*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(4*  \
                    k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)  \
                    **2)**0.5E0)**(-3)+(-4)*k**2*P0*T0**(-2)*Ta*V0*((-0.1962E2)  \
                    *M+(-1)*P0*S+k*S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*V0+(  \
                    0.1962E2*M+P0*S+(-1)*k*S**(-1)*V0)**2)**0.5E0)**(-2))*(M*(  \
                    k+4*k**2*P0*T0**(-1)*Ta*V0*((-0.1962E2)*M+(-1)*P0*S+k*  \
                    S**(-1)*V0+(4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*  \
                    k*S**(-1)*V0)**2)**0.5E0)**(-2))**(-1))**(-0.5E0)*(k+4*k**2*  \
                    P0*T0**(-1)*Ta*V0*((-0.1962E2)*M+(-1)*P0*S+k*S**(-1)*V0+(  \
                    4*k*P0*T0**(-1)*Ta*V0+(0.1962E2*M+P0*S+(-1)*k*S**(-1)*  \
                    V0)**2)**0.5E0)**(-2))**(-2)]) 
    scaling = np.array([30,0.02-.005,0.01-0.002,4000,20000,6,20])*0.5
    dC = out*scaling
    return [C,dC]
