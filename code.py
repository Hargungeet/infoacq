

import numpy as np
import scipy 
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import fsolve, fminbound
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
xgrid= 75


r = 4
alpha= 0.4
cost= 0.1

j=0

x_span= np.linspace(-0.8, 0.8, xgrid)



zscore= np.random.normal(loc=0, scale=1,size= 1000)

def intgrl(xi,x):
	return -np.exp(-r*xi)*(2*x**2/(1-4*xi)-np.log(np.sqrt(1-4*xi)))


def parmup(parms):
	
	sigma = np.where(parms[0:xgrid]<0, np.zeros(xgrid), parms[0:xgrid])
	#sigma= parms[0:xgrid]
	#sigma= np.abs(parms[0:xgrid])
	c_val= -np.abs(parms[xgrid:])
	parmnew =np.ones(2*xgrid)
	v0= np.ones(xgrid)
	meanv = np.ones(xgrid)
	v_exp=np.ones(xgrid)
	snew= np.ones(xgrid) 
	cnew = -np.ones(xgrid)
	
	for i in range(xgrid):
		v0[i]=  (c_val[i] )# +np.exp(x_span[i]**2 ) /(1-r))
	#print(v0)
	v_guess1 = interp1d(x_span, v0, fill_value="extrapolate") 
	def v_guess(x) : return  np.amax([v_guess1(x), np.amax(v0)-alpha ])
	
	
	# if v_guess(0)<np.amax(v0):
	# 	i=0
	# 	for i in range(xgrid):
	# 		v0[i]=  v_guess(0)# +np.exp(x_span[i]**2 ) /(1-r))
	# 	#print(v0)
	# 	v_guess1 = interp1d(x_span, v0, fill_value="extrapolate") 
	# 	def v_guess(x) : return  np.amax([v_guess1(x), np.amax(v0)-alpha ])
	
	
	

	i=0
	
	while i<xgrid:
		x= x_span[i]
		def parm_update(parms2):
			s= parms2[0]
			c= parms2[1]

			
			z_val= x+np.sqrt(np.abs(s))*zscore
			v= [v_guess(z) for z in z_val]

			v_exp[i] = np.mean([v_guess(z)*(z-x)**2 for z in z_val]  ) # E(v(z)(x-z)^2)
			
			meanv[i]=np.mean(v)
			# def expec(z):
			# 	return v_guess(z)*norm.pdf(z, x, np.sqrt(s))
			# if s==0:
			# 	meanv1= np.zeros(2)
			# 	meanv1[0] = v_guess(x)
			# else:
			# 	meanv1=quad( expec, -np.inf, np.inf )
			# meanv[i]= meanv1[0]

			# def dev_exp(z):
			# 	return v_guess(z)*(z-x)**2*norm.pdf(z, x, np.sqrt(s))

			# if s==0:
			# 	meanvexp1= np.zeros(2)
			# 	meanvexp1[0] = 0
			# else:
			# 	meanvexp1=quad( dev_exp, -np.inf, np.inf )

			# v_exp[i] =meanvexp1[0]
			vs=(meanv[i]- cost)
			v_dev= (-meanv[i]/(2*s))+1/(2*s**2)*v_exp[i]
			#if s==0:
			#	v_dev=0
			
			def tosolve(xi):
				
				
				return (r *vs+(2*x**2/(1-4*xi))+np.log(1/np.sqrt(1-4*xi))-v_dev)**2

			snew[i] = (fminbound(tosolve, 0, 0.25))# Due to V'_s=0 rV=integral
			
			defint=quad(intgrl, 0, snew[i], args=(x))
			
			cnew[i]= vs*np.exp(-r*snew[i])+defint[0] #
			#val_match= (c *np.exp(r* s) - (r *(s+ x**2)  + 1)/r**2) - (np.mean(v)-cost)
			#smooth_cond= (c *r*np.exp(r* s) - 1/r)
			
			
			return [ snew[i] ,cnew[i]]

		parmroot = parm_update( [sigma[i], c_val[i]])#, bounds=([0, None], [None, None]))
		parmnew[i] =parmroot[0]
		parmnew[xgrid+i]=parmroot[1] 
		
		#print(v[i])
		#print(v[1]==v[0])
		#print(np.mean(v[1] )==np.mean(v[0] ))
		
		
		i=i+1
	
	return parmnew


parms = np.ones(2*xgrid)

parm_up=parms*0.15
sigma= np.abs(parms[0:xgrid])
c_val= parms[xgrid:]
tol= 0.004

while np.max(np.abs(parms-parm_up))>tol :# or np.max( np.diff(sigma))>3:
	
	parms =parm_up
	#if np.max( np.diff(sigma))>1:
	#	parms[np.argmax(np.diff(sigma))]=parms[np.argmax(np.diff(sigma))-1]
	# if np.max( np.diff(c_val))>1:
	# 	parms[np.argmax(np.diff(sigma))]=parms[np.argmax(np.diff(sigma))-1]
	sigmatemp= interp1d(x_span, np.abs(parms[0:xgrid]), fill_value="extrapolate")
	c_val= parms[xgrid:]
	parm_up = parmup(parms)
	#inter_val[j]=[ (c_val  - (r *(x_span**2 ) + 1)/r**2) ]
	print(j)
	print(np.max(np.abs(parms-parm_up)))
	j=j+1
	#print(parms, parm_up)
	#print(np.max(np.abs(parms-parm_up)))
print('done')

v0 = (c_val )
#print(v0)
v_guess1 = interp1d(x_span, v0, fill_value="extrapolate") 
v_guess = lambda x: np.amax([v_guess1(x), np.amax(v0)-alpha ])

j=0
lim= np.ones(2)

while j<xgrid-1:
	if v_guess(x_span[j])==v_guess1(0)-alpha and v_guess(x_span[j])<v_guess(x_span[j+1]):
		lim[0] = x_span[j]
	if v_guess(x_span[j])==v_guess1(0)-alpha and v_guess(x_span[j])<v_guess(x_span[j-1]):
		lim[1] = x_span[j]
	j=j+1

val = [v_guess(x) for x in x_span]

plt.figure()
plt.plot(x_span, val)
plt.xlabel('x')
plt.ylabel('v0')

plt.savefig('v0.pgf')
def sigma(x):
	if x<=lim[0] or x>=lim[1]:
		return 0
	else:
		return sigmatemp(x)

plt.figure() 

plt.plot(x_span, [sigma(x) for x in x_span])

# plt.plot(x_span[0:lim[0]], np.zeros(lim[0]), '-b')
# plt.plot( x_span[lim[0]:lim[1]],[sigmatemp(x) for x in x_span[lim[0]:lim[1]]], '-b')
# plt.plot( x_span[lim[1]:], [sigma(x) for x in x_span[lim[1]:]], '-b' )
plt.xlabel('x') 
plt.ylabel('sigma') 
plt.show()
plt.savefig('sigma.pgf')