from refraction_render.atms import std_atmosphere

def superior_mirage(h):
	return 1.2*np.exp(-((h-23)/3)**2)

def inferior_mirage(h):
	e = np.nan_to_num(np.exp(h/0.01))
	return (2/(1+e))*0.05

def soundly(h):
	e = np.nan_to_num(np.exp(h/0.01))
	return (2/(1+e))*0.05

def isle_of_man(h):
	e1 = np.exp(h/1.5)
	e2 = np.exp(h/0.1)
	return (2/(1+e1))*0.1+(2/(1+e2))*0.15


atm_args = dict(T_prof=superior_mirage)
atm = std_atmosphere(**atm_args)


atm_args = dict(T_prof=inferior_mirage)
atm = std_atmosphere(**atm_args)


atm_args = dict(T_prof=soundly)
atm = std_atmosphere(**atm_args)


atm_args = dict(T0=8.3,P0=103000,T_prof=isle_of_man)
atm = std_atmosphere(**atm_args)