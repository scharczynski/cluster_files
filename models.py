import numpy as np
from maxlikespy.model import Model
import autograd.numpy as np
import autograd.scipy.special as sse
import matplotlib.pyplot as plt
# from scipy import special as sse


class Time(Model):

    """Model which contains a time dependent gaussian compenent 
    and an offset parameter.

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    num_params : int
        Integer signifying the number of model parameters.
    spikes : dict     
        Dict containing binned spike data for current cell.

    """

    def __init__(self, data):
        super().__init__(data)
        # self.spikes = data['spikes']
        self.param_names = ["a_1", "ut", "st", "a_0"]
        # self.x0 = [1e-5, 100, 100, 1e-5]


    def objective(self, x):
        fun = self.model(x)
        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def model(self, x, plot=False):
        a, ut, st, o = x

        self.function = (
            (a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        return self.function

    def plot_model(self, x):
        a, ut, st, o = x
        self.function = (
            (a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        return self.function

class SigmaMuTau(Model):

    def __init__(self, data):
        super().__init__(data)
        # self.spikes = data['spikes']
        self.param_names = ["sigma", "mu", "tau", "a_1", "a_0"]
        # self.x0 = [100, 5000, 0.001, 1e-5, 1e-5]


    def model(self, x):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''

        s, mu, tau, a_1, a_0 = x
        l = 1/tau
        '''old method'''
        # fun = a_1*np.exp(-0.5*(np.power((self.t-m)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-m)/s))))
        # ) + a_0

        fun = (a_1*(np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x)
        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def plot_model(self, x):
        return self.model(x)


class SigmaMuTauStimRP(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu","tau", "a_1", "a_2", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

       
    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 2))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["pair_stim_class"]
            if stim_class == '0':
                stim_matrix[int(trial_num)][:] = [1, 0]
            elif stim_class == '1':
                stim_matrix[int(trial_num)][:] = [1, 0]
            elif stim_class == '2':
                stim_matrix[int(trial_num)][:] = [0, 1]
            elif stim_class == '3':
                stim_matrix[int(trial_num)][:] = [0, 1]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''
        s, mu, tau, a_1,a_2, a_0 = x
        l = 1/tau
        fun1 = (np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
            + (a_2*(self.stim_matrix[:, 1] *fun1.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def plot_model(self, x):
        s, mu, tau, a_1,a_2, a_0 = x
        print("final fit in plot {0}".format(x))

        l = 1/tau

        fun1 = (np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))


        fun = (
            a_1*(self.stim_matrix[:, 0] * fun1.T) 
            + (a_2*(self.stim_matrix[:, 1] *fun1.T))
        ) + a_0

        return (np.sum(fun, axis=1)/fun.shape[1])


class SigmaMuTauStimClassRP(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = [
            "sigma", 
            "mu",
            "tau", 
            "a_1", 
            "a_2", 
            "a_3",
            "a_4",
            "a_0"
        ]
        self.t = np.tile(self.t, (self.num_trials, 1))

       
    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["pair_stim_class"]
            if stim_class == '0':
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            elif stim_class == '1':
                stim_matrix[int(trial_num)][:] = [0, 1, 0, 0]
            elif stim_class == '2':
                stim_matrix[int(trial_num)][:] = [0, 0, 1, 0]
            elif stim_class == '3':
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 1]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''
        s, mu, tau, a_1,a_2,a_3, a_4, a_0 = x
        l = 1/tau
        fun1 = (np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
            + (a_2*(self.stim_matrix[:, 1] *fun1.T))
            + (a_3*(self.stim_matrix[:, 2] *fun1.T))
            + (a_4*(self.stim_matrix[:, 3] *fun1.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)


        return (np.sum(fun, axis=1)/fun.shape[1])

class SigmaMuTauStimWarden(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu","tau", "a_1", "a_2","a_3", "a_4", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 1:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            elif stim_class == 2:
                stim_matrix[int(trial_num)][:] = [0, 1, 0, 0]
            elif stim_class == 3:
                stim_matrix[int(trial_num)][:] = [0, 0, 1, 0]
            elif stim_class == 4:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 1]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        s, mu, tau, a_1,a_2,a_3, a_4, a_0 = x
        l = 1/tau
        fun1 = (np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
            + (a_2*(self.stim_matrix[:, 1] *fun1.T))
            + (a_3*(self.stim_matrix[:, 2] *fun1.T))
            + (a_4*(self.stim_matrix[:, 3] *fun1.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)


        return (np.sum(fun, axis=1)/fun.shape[1])

class SigmaMuTauDual(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma1","sigma2", "mu1", "mu2", "tau1","tau2", "a_1", "a_0"]
        # self.t = np.tile(self.t, (self.num_trials, 1))

       
    # def info_callback(self):
    #     self.stims = self.info["stim_identity"]
    #     stim_matrix = np.zeros((self.spikes.shape[0], 4))
    #     for trial, stim in enumerate(self.stims):
    #         if stim == '0':
    #             stim_matrix[trial][:] = [1, 0, 1, 0]
    #         elif stim == '1':
    #             stim_matrix[trial][:] = [1, 0, 0, 1]
    #         elif stim == '2':
    #             stim_matrix[trial][:] = [0, 1, 1, 0]
    #         elif stim == '3':
    #             stim_matrix[trial][:] = [0, 1, 0, 1]
    #     self.stim_matrix = stim_matrix

    def model(self, x, plot=False):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''
        s1,s2, mu1, mu2, tau1, tau2, a_1, a_0 = x
        print(x)
        # fun1 = a_1*np.exp(-0.5*(np.power((self.t-mu1)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-mu1)/s))))
        # )
        # fun2 = a_2*np.exp(-0.5*(np.power((self.t-mu2)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-mu2)/s))))
        # )
        l1 = 1/tau1
        l2 = 1/tau2
        fun1 = a_1*(np.exp((l1/2)*(2*mu1+l1*s1**2-2*self.t))*sse.erfc((mu1+l1*s1**2-self.t)/(np.sqrt(2)*s1)))
        fun2 = a_1*(np.exp((l2/2)*(2*mu2+l2*s2**2-2*self.t))*sse.erfc((mu2+l2*s2**2-self.t)/(np.sqrt(2)*s2)))


        # if any(np.isnan(fun1)):
        #     print(np.where(np.isnan(fun1)))
        return fun1 + fun2 + a_0

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def plot_model(self, x):
        s1, s2, mu1, mu2, tau1, tau2, a_1, a_0 = x
        print("final fit in plot {0}".format(x))
        # fun1 = a_1*np.exp(-0.5*(np.power((self.t[0]-mu1)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t[0]-mu1)/s))))
        # )
        # fun2 = a_2*np.exp(-0.5*(np.power((self.t[0]-mu2)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t[0]-mu2)/s))))
        # )
        l1 = 1/tau1
        l2 = 1/tau2
        fun1 = a_1*(np.exp((l1/2)*(2*mu1+l1*s1**2-2*self.t))*sse.erfc((mu1+l1*s1**2-self.t)/(np.sqrt(2)*s1)))
        fun2 = a_1*(np.exp((l2/2)*(2*mu2+l2*s2**2-2*self.t))*sse.erfc((mu2+l2*s2**2-self.t)/(np.sqrt(2)*s2)))
        # fun1 = a_1*(l/2 * np.exp((l/2)*(2*mu1+l*s**2-2*self.t[0]))*sse.erfc((mu1+l*s**2-self.t[0])/np.sqrt(2)*s))
        # fun2 = a_2*(l/2 * np.exp((l/2)*(2*mu2+l*s**2-2*self.t[0]))*sse.erfc((mu2+l*s**2-self.t[0])/np.sqrt(2)*s))

        return fun1 + fun2 + a_0

class SigmaMuTauStim(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu","tau", "a_1", "a_2","a_3", "a_4", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

       
    def info_callback(self):
        self.stims = self.info["stim_identity"]
        '''for warden'''
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        self.t = self.t[list(map(int, trials))]
        for trial_num, trial in enumerate(trials):
            samples = self.stims[(trial)]["sample"]
            if samples[0] == 1:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            elif samples[0] == 2:
                stim_matrix[int(trial_num)][:] = [0, 1, 0, 0]
            elif samples[0] == 3:
                stim_matrix[int(trial_num)][:] = [0, 0, 1, 0]
            elif samples[0] == 4:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 1]
        print("what")
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''
        s, mu, tau, a_1,a_2,a_3, a_4, a_0 = x
        l = 1/tau
        fun1 = (np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
            + (a_2*(self.stim_matrix[:, 1] *fun1.T))
            + (a_3*(self.stim_matrix[:, 2] *fun1.T))
            + (a_4*(self.stim_matrix[:, 3] *fun1.T))
        ) + a_0

        '''for warden'''

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def plot_model(self, x):
        s, mu, tau, a_1,a_2,a_3, a_4, a_0 = x
        print("final fit in plot {0}".format(x))

        l = 1/tau

        fun1 = (np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))


        fun = (
            a_1*(self.stim_matrix[:, 0] * fun1.T) 
            + (a_2*(self.stim_matrix[:, 1] *fun1.T))
            + (a_3*(self.stim_matrix[:, 2] *fun1.T))
            + (a_4*(self.stim_matrix[:, 3] *fun1.T))
        ) + a_0

        return (np.sum(fun, axis=1)/fun.shape[1])


class SigmaMuTauDualStim(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma1","sigma2", "mu1", "mu2", "tau1","tau2", "a_1", "a_2","a_3", "a_4", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

       
    def info_callback(self):
        self.stims = self.info["stim_identity"]
        '''for warden'''
        stim_matrix1p = np.zeros((self.spikes.shape[0], 4))
        stim_matrix2p = np.zeros((self.spikes.shape[0], 4))
        for trial in self.stims:
            samples = self.stims[trial]["sample"]
            if samples[0] == 1:
                stim_matrix1p[int(trial)][:] = [1, 0, 0, 0]
            elif samples[0] == 2:
                stim_matrix1p[int(trial)][:] = [0, 1, 0, 0]
            elif samples[0] == 3:
                stim_matrix1p[int(trial)][:] = [0, 0, 1, 0]
            elif samples[0] == 4:
                stim_matrix1p[int(trial)][:] = [0, 0, 0, 1]
            if samples[1] == 1:
                stim_matrix2p[int(trial)][:] = [1, 0, 0, 0]
            elif samples[1] == 2:
                stim_matrix2p[int(trial)][:] = [0, 1, 0, 0]
            elif samples[1] == 3:
                stim_matrix2p[int(trial)][:] = [0, 0, 1, 0]
            elif samples[1] == 4:
                stim_matrix2p[int(trial)][:] = [0, 0, 0, 1]


        self.stim_matrix1p = stim_matrix1p
        self.stim_matrix2p = stim_matrix2p


        # '''for rossi pool'''
        # stim_matrix = np.zeros((self.spikes.shape[0], 4))
        # for trial, stim in enumerate(self.stims):
        #     if stim == '0':
        #         stim_matrix[trial][:] = [1, 0, 1, 0] #GG
        #     elif stim == '1':
        #         stim_matrix[trial][:] = [1, 0, 0, 1] #GE
        #     elif stim == '2':
        #         stim_matrix[trial][:] = [0, 1, 1, 0] #EG
        #     elif stim == '3':
        #         stim_matrix[trial][:] = [0, 1, 0, 1] #EE
        # self.stim_matrix = stim_matrix

    def model(self, x, plot=False):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''
        s1,s2, mu1, mu2, tau1, tau2, a_1,a_2,a_3, a_4, a_0 = x
        print(x)
        # fun1 = a_1*np.exp(-0.5*(np.power((self.t-mu1)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-mu1)/s))))
        # )
        # fun2 = a_2*np.exp(-0.5*(np.power((self.t-mu2)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-mu2)/s))))
        # )
        l1 = 1/tau1
        l2 = 1/tau2
        fun1 = (np.exp((l1/2)*(2*mu1+l1*s1**2-2*self.t))*sse.erfc((mu1+l1*s1**2-self.t)/(np.sqrt(2)*s1)))
        fun2 = (np.exp((l2/2)*(2*mu2+l2*s2**2-2*self.t))*sse.erfc((mu2+l2*s2**2-self.t)/(np.sqrt(2)*s2)))

        '''for rossi pool'''
        fun = (
            a_1*(self.stim_matrix1p[:, 0] * fun1.T) 
            + (a_2*(self.stim_matrix1p[:, 1] *fun1.T))
            + (a_3*(self.stim_matrix1p[:, 2] *fun1.T))
            + (a_4*(self.stim_matrix1p[:, 3] *fun1.T))
        ) + (
            a_1*(self.stim_matrix2p[:, 0] * fun2.T) 
            + (a_2*(self.stim_matrix2p[:, 1] *fun2.T))
            + (a_3*(self.stim_matrix2p[:, 2] *fun2.T))
            + (a_4*(self.stim_matrix2p[:, 3] *fun2.T))
        ) + a_0
        # self.stim_matrix1p[:, 0] = a_1*self.stim_matrix1p[:, 0]
        # self.stim_matrix1p[:, 1] = a_2*self.stim_matrix1p[:, 1]
        # self.stim_matrix1p[:, 2] = a_3*self.stim_matrix1p[:, 2]
        # self.stim_matrix1p[:, 3] = a_4*self.stim_matrix1p[:, 3]
        # self.stim_matrix2p[:, 0] = a_1*self.stim_matrix2p[:, 0]
        # self.stim_matrix2p[:, 1] = a_2*self.stim_matrix2p[:, 1]
        # self.stim_matrix2p[:, 2] = a_3*self.stim_matrix2p[:, 2]
        # self.stim_matrix2p[:, 3] = a_4*self.stim_matrix2p[:, 3]  
        # fun = (self.stim_matrix1p * fun1.T) + (self.stim_matrix2p * fun2.T) + a_0
        '''for warden'''

        # if any(np.isnan(fun1)):
        #     print(np.where(np.isnan(fun1)))
        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def plot_model(self, x):
        s1, s2, mu1, mu2, tau1, tau2, a_1,a_2,a_3, a_4, a_0 = x
        print("final fit in plot {0}".format(x))
        # fun1 = a_1*np.exp(-0.5*(np.power((self.t[0]-mu1)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t[0]-mu1)/s))))
        # )
        # fun2 = a_2*np.exp(-0.5*(np.power((self.t[0]-mu2)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t[0]-mu2)/s))))
        # # )
        l1 = 1/tau1
        l2 = 1/tau2
        fun1 = (np.exp((l1/2)*(2*mu1+l1*s1**2-2*self.t))*sse.erfc((mu1+l1*s1**2-self.t)/(np.sqrt(2)*s1)))
        fun2 = (np.exp((l2/2)*(2*mu2+l2*s2**2-2*self.t))*sse.erfc((mu2+l2*s2**2-self.t)/(np.sqrt(2)*s2)))

        # fun = (a_1*((self.stim_matrix[:, 0] * fun1.T) + (self.stim_matrix[:,2] *fun2.T)) + (    
        #     (a_2*((self.stim_matrix[:, 1] * fun1.T) + (self.stim_matrix[:,3] *fun2.T))) + a_0))

        fun = (
            a_1*(self.stim_matrix1p[:, 0] * fun1.T) 
            + (a_2*(self.stim_matrix1p[:, 1] *fun1.T))
            + (a_3*(self.stim_matrix1p[:, 2] *fun1.T))
            + (a_4*(self.stim_matrix1p[:, 3] *fun1.T))
        ) + (
            a_1*(fun2.T) 
            + (a_2*(fun2.T))
            + (a_3*(fun2.T))
            + (a_4*(fun2.T))
        ) + a_0
        # l1 = 1/tau1
        # l2 = 1/tau2
        # fun1 = a_1*(np.exp((l1/2)*(2*mu1+l1*s1**2-2*self.t[0]))*sse.erfc((mu1+l1*s1**2-self.t[0])/(np.sqrt(2)*s1)))
        # fun2 = a_2*(np.exp((l2/2)*(2*mu2+l2*s2**2-2*self.t[0]))*sse.erfc((mu2+l2*s2**2-self.t[0])/(np.sqrt(2)*s2)))
        # fun1 = a_1*(l/2 * np.exp((l/2)*(2*mu1+l*s**2-2*self.t[0]))*sse.erfc((mu1+l*s**2-self.t[0])/np.sqrt(2)*s))
        # fun2 = a_2*(l/2 * np.exp((l/2)*(2*mu2+l*s**2-2*self.t[0]))*sse.erfc((mu2+l*s**2-self.t[0])/np.sqrt(2)*s))

        return (np.sum(fun, axis=1)/fun.shape[1])

class SigmaMuTauDualStimPos(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu1", "mu2", "tau", "a_1", "a_2","a_3","a_4", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

       
    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        for trial, stim in enumerate(self.stims):
            if stim == '0':
                stim_matrix[trial][:] = [1, 0, 1, 0]
            elif stim == '1':
                stim_matrix[trial][:] = [1, 0, 0, 1]
            elif stim == '2':
                stim_matrix[trial][:] = [0, 1, 1, 0]
            elif stim == '3':
                stim_matrix[trial][:] = [0, 1, 0, 1]
        self.stim_matrix = stim_matrix


    def erfcx(self,x):

        if np.isscalar(x) ==  1:
            if x < 25:
                return sse.erfc(x) * np.exp(x*x)
            else:
                y = 1. / x
                z = y * y
                s = y*(1.+z*(-0.5+z*(0.75+z*(-1.875+z*(6.5625-29.53125*z)))))
                return s * 0.564189583547756287
        else:
            over_ind = np.where(x>=25)[0]
            good_ind = np.where(x<25)[0]
            
            x[good_ind] = sse.erfc(x[good_ind]) * np.exp(x[good_ind]*x[good_ind])
            y = 1. / x[over_ind]
            z = y * y
            s = y*(1.+z*(-0.5+z*(0.75+z*(-1.875+z*(6.5625-29.53125*z)))))
            x[over_ind] = s * 0.564189583547756287
            return x

    def model(self, x, plot=False):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''
        s, mu1, mu2, tau, a_1,a_2, a_3, a_4, a_0 = x
        print(x)
        # fun1 = a_1*np.exp(-0.5*(np.power((self.t-mu1)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-mu1)/s))))
        # )
        # fun2 = a_2*np.exp(-0.5*(np.power((self.t-mu2)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-mu2)/s))))
        # )
        l = 1/tau
        fun1 = a_1*(l/2 * np.exp((l/2)*(2*mu1+l*s**2-2*self.t))*sse.erfc((mu1+l*s**2-self.t)/np.sqrt(2)*s))
        fun2 = a_2*(l/2 * np.exp((l/2)*(2*mu2+l*s**2-2*self.t))*sse.erfc((mu2+l*s**2-self.t)/np.sqrt(2)*s))
        fun3 = a_3*(l/2 * np.exp((l/2)*(2*mu1+l*s**2-2*self.t))*sse.erfc((mu1+l*s**2-self.t)/np.sqrt(2)*s))
        fun4 = a_4*(l/2 * np.exp((l/2)*(2*mu2+l*s**2-2*self.t))*sse.erfc((mu2+l*s**2-self.t)/np.sqrt(2)*s))

        fun = (self.stim_matrix[:, 0] * fun1.T) + (
            (self.stim_matrix[:, 1] * fun3.T) +
            (self.stim_matrix[:, 2] * fun2.T) +
            (self.stim_matrix[:, 3] * fun4.T)
        ) + a_0
        if any(np.isnan(fun[:,0])):
            print(np.where(np.isnan(fun)))
        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def plot_model(self, x):
        s, mu1, mu2, tau, a_1,a_2,a_3, a_4, a_0 = x

        # fun1 = a_1*np.exp(-0.5*(np.power((self.t[0]-mu1)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t[0]-mu1)/s))))
        # )
        # fun2 = a_2*np.exp(-0.5*(np.power((self.t[0]-mu2)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t[0]-mu2)/s))))
        # )
        l = 1/tau
        fun1 = a_1*(l/2 * np.exp((l/2)*(2*mu1+l*s**2-2*self.t[0]))*sse.erfc((mu1+l*s**2-self.t[0])/np.sqrt(2)*s))
        fun2 = a_2*(l/2 * np.exp((l/2)*(2*mu2+l*s**2-2*self.t[0]))*sse.erfc((mu2+l*s**2-self.t[0])/np.sqrt(2)*s))
        fun3 = a_3*(l/2 * np.exp((l/2)*(2*mu1+l*s**2-2*self.t[0]))*sse.erfc((mu1+l*s**2-self.t[0])/np.sqrt(2)*s))
        fun4 = a_4*(l/2 * np.exp((l/2)*(2*mu2+l*s**2-2*self.t[0]))*sse.erfc((mu2+l*s**2-self.t[0])/np.sqrt(2)*s))

        return fun1 + fun2 + fun3 + fun4 + a_0


# class SigmaMuTauStim(Model):
#     def __init__(self, data):
#         super().__init__(data)
#         self.param_names = ["sigma", "mu", "tau", "a_1", "a_2", "a_0"]
#         self.t = np.tile(self.t, (self.num_trials, 1))

       
#     def info_callback(self):
#         self.stims = self.info["stim_identity"]
#         stim_matrix = np.zeros((self.spikes.shape[0], 4))
#         for trial, stim in enumerate(self.stims):
#             if stim == '0':
#                 stim_matrix[trial][:] = [1, 0, 1, 0]
#             elif stim == '1':
#                 stim_matrix[trial][:] = [1, 0, 0, 1]
#             elif stim == '2':
#                 stim_matrix[trial][:] = [0, 1, 1, 0]
#             elif stim == '3':
#                 stim_matrix[trial][:] = [0, 1, 0, 1]
#         self.stim_matrix = stim_matrix


#     def erfcx(self,x):

#         if np.isscalar(x) ==  1:
#             if x < 25:
#                 return sse.erfc(x) * np.exp(x*x)
#             else:
#                 y = 1. / x
#                 z = y * y
#                 s = y*(1.+z*(-0.5+z*(0.75+z*(-1.875+z*(6.5625-29.53125*z)))))
#                 return s * 0.564189583547756287
#         else:
#             over_ind = np.where(x>=25)[0]
#             good_ind = np.where(x<25)[0]
            
#             x[good_ind] = sse.erfc(x[good_ind]) * np.exp(x[good_ind]*x[good_ind])
#             y = 1. / x[over_ind]
#             z = y * y
#             s = y*(1.+z*(-0.5+z*(0.75+z*(-1.875+z*(6.5625-29.53125*z)))))
#             x[over_ind] = s * 0.564189583547756287
#             return x

#     def model(self, x, plot=False):
#         '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
#         '''
#         s, mu, tau, a_1, a_2, a_0 = x
#         print(x)
#         # fun1 = a_1*np.exp(-0.5*(np.power((self.t-mu1)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
#         #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-mu1)/s))))
#         # )
#         # fun2 = a_2*np.exp(-0.5*(np.power((self.t-mu2)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
#         #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-mu2)/s))))
#         # )
#         l = 1/tau
#         fun1 = a_1*(l/2 * np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/np.sqrt(2)*s))
#         fun2 = a_2*(l/2 * np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/np.sqrt(2)*s))
#         fun = (self.stim_matrix[:, 0] * fun1.T) + (
#             (self.stim_matrix[:, 1] * fun2.T) +
#             (self.stim_matrix[:, 2] * fun1.T) +
#             (self.stim_matrix[:, 3] * fun2.T)
#         ) + a_0

#         if any(np.isnan(fun[:,0])):
#             print(np.where(np.isnan(fun)))
#         return fun

#     def objective(self, x):
#         fun = self.model(x).T

#         obj = np.sum(self.spikes * (-np.log(fun)) +
#                       (1 - self.spikes) * (-np.log(1 - (fun))))
        
#         return obj

#     def plot_model(self, x):
#         s, mu, tau, a_1, a_2, a_0 = x

#         # fun1 = a_1*np.exp(-0.5*(np.power((self.t[0]-mu1)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
#         #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t[0]-mu1)/s))))
#         # )
#         # fun2 = a_2*np.exp(-0.5*(np.power((self.t[0]-mu2)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
#         #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t[0]-mu2)/s))))
#         # )
#         l = 1/tau
#         fun1 = a_1*(l/2 * np.exp((l/2)*(2*mu+l*s**2-2*self.t[0]))*sse.erfc((mu+l*s**2-self.t[0])/np.sqrt(2)*s))
#         fun2 = a_2*(l/2 * np.exp((l/2)*(2*mu+l*s**2-2*self.t[0]))*sse.erfc((mu+l*s**2-self.t[0])/np.sqrt(2)*s))
#         return fun1 + fun2 + a_0


class Const(Model):

    """Model which contains only a single offset parameter.

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    num_params : int
        Integer signifying the number of model parameters.

    """

    def __init__(self, data):
        super().__init__(data)
        # self.spikes = data['spikes']
        self.param_names = ["a_0"]
        # self.x0 = [0.1]

    def model(self, x, plot=False):
        o = x
        return o

    def objective(self, x):
        fun = self.model(x)
        obj = (np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun)))))
        return obj

    def pso_con(self, x):
        return 1 - x

class SigmaMuTauNone(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu", "tau", "a_1", "a_0"]
        # self.t = np.tile(self.t, (self.num_trials, 1))

       
    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        for trial, stim in enumerate(self.stims):
            if stim == '0':
                stim_matrix[trial][:] = [1, 0, 1, 0]
            elif stim == '1':
                stim_matrix[trial][:] = [1, 0, 0, 1]
            elif stim == '2':
                stim_matrix[trial][:] = [0, 1, 1, 0]
            elif stim == '3':
                stim_matrix[trial][:] = [0, 1, 0, 1]
        self.stim_matrix = stim_matrix


    # def erfcx(self,x):

    #     if np.isscalar(x) ==  1:
    #         if x < 25:
    #             return sse.erfc(x) * np.exp(x*x)
    #         else:
    #             y = 1. / x
    #             z = y * y
    #             s = y*(1.+z*(-0.5+z*(0.75+z*(-1.875+z*(6.5625-29.53125*z)))))
    #             return s * 0.564189583547756287
    #     else:
    #         over_ind = np.where(x>=25)[0]
    #         good_ind = np.where(x<25)[0]
            
    #         x[good_ind] = sse.erfc(x[good_ind]) * np.exp(x[good_ind]*x[good_ind])
    #         y = 1. / x[over_ind]
    #         z = y * y
    #         s = y*(1.+z*(-0.5+z*(0.75+z*(-1.875+z*(6.5625-29.53125*z)))))
    #         x[over_ind] = s * 0.564189583547756287
    #         return x

    def model(self, x, plot=False):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''
        s, mu, tau, a_1, a_0 = x
        print(x)
        # fun1 = a_1*np.exp(-0.5*(np.power((self.t-mu1)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-mu1)/s))))
        # )
        # fun2 = a_2*np.exp(-0.5*(np.power((self.t-mu2)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-mu2)/s))))
        # )
        l = 1/tau
        fun = a_1*(l/2 * np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/np.sqrt(2)*s))
        # fun2 = a_2*(l/2 * np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/np.sqrt(2)*s))
        # fun = self.stim_matrix[:, 0] * fun1.T + (self.stim_matrix[:, 2] * fun1.T) + a_0

        return fun + a_0

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def plot_model(self, x):
        s, mu, tau, a_1, a_0 = x

        # fun1 = a_1*np.exp(-0.5*(np.power((self.t[0]-mu1)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t[0]-mu1)/s))))
        # )
        # fun2 = a_2*np.exp(-0.5*(np.power((self.t[0]-mu2)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t[0]-mu2)/s))))
        # )
        l = 1/tau
        fun1 = a_1*(l/2 * np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/np.sqrt(2)*s))
        # fun2 = a_2*(l/2 * np.exp((l/2)*(2*mu+l*s**2-2*self.t[0]))*sse.erfc((mu+l*s**2-self.t[0])/np.sqrt(2)*s))
        return fun1 + a_0


class CatSetTime(Model):

    """Model which contains seperate time-dependent gaussian terms per each given category sets.

    Parameters
    ----------
    time_params : list
        List of gaussian parameters from a previous time-only fit.
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.

    Attributes
    ----------
    t : numpy.ndarray
        Array of timeslices of size specified by time_low, time_high and time_bin.
        This array is repeated a number of times equal to the amount of trials
        this cell has.
    name : string
        Human readable string describing the model.
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.
    ut : float
        Mean of gaussian distribution.
    st : float
        Standard deviation of gaussian distribution.
    a1 : float
        Coefficient of category set 1 gaussian distribution.
    a2 : float
        Coefficient of category set 2 gaussian distribution.
    o : float
        Additive offset of distribution.

    """

    def __init__(self, data):
        super().__init__(data)
        self.plot_t = self.t
        self.t = np.tile(self.t, (data["num_trials"], 1))
        self.conditions = data["conditions"]
        self.param_names = ["ut", "st", "a_0", "a_1", "a_2"]
        self.x0 = [500, 100, 0.001, 0.001, 0.001]
        self.spikes = data["spikes"]
        self.info = {}
        self.xform_cond = None

    def model(self, x, plot=False, condition=None):
        pairs = self.info["pairs"]
        ut, st, a_0 = x[0], x[1], x[2]
        a_1, a_2 = x[3], x[4]
        pair_1 = pairs[0]
        pair_2 = pairs[1]
        if not self.xform_cond:
            self.xform_cond = {}
            self.xform_cond[1] = self.conditions[pair_1[0]] + self.conditions[pair_1[1]]
            self.xform_cond[2] = self.conditions[pair_2[0]] + self.conditions[pair_2[1]]
            self.conditions = self.xform_cond
        c1 = self.conditions[1]
        c2 = self.conditions[2]
        # c1 = self.conditions[pair_1[0]] + self.conditions[pair_1[1]]
        # c2 = self.conditions[pair_2[0]] + self.conditions[pair_2[1]]

        if plot:
            if condition==1:
                return ((a_1  * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.)))))
            elif condition==2:
                return ((a_2  * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.)))))
            else:
                return ((a_1  * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.)))) + \
                (a_2 * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.))))) + a_0

        return ((a_1 * c1 * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + \
            (a_2 * c2 * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.))))) + a_0

    def objective(self, x):
        fun = self.model(x)
        return np.sum((self.spikes * (-np.log(fun)) +
                        (1 - self.spikes) * (-np.log(1 - (fun)))))

class CatTime(Model):

    """Model which contains seperate time-dependent gaussian terms per each given category.

    Parameters
    ----------
    time_params : list
        List of gaussian parameters from a previous time-only fit.
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.

    Attributes
    ----------
    t : numpy.ndarray
        Array of timeslices of size specified by time_low, time_high and time_bin.
        This array is repeated a number of times equal to the amount of trials
        this cell has.
    name : string
        Human readable string describing the model.
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.
    ut : float
        Mean of gaussian distribution.
    st : float
        Standard deviation of gaussian distribution.
    a1 : float
        Coefficient of category 1 gaussian distribution.
    a2 : float
        Coefficient of category 2 gaussian distribution.
    a3 : float
        Coefficient of category 3 gaussian distribution.
    a4 : float
        Coefficient of category 4 gaussian distribution.
    o : float
        Additive offset of distribution.

    """

    def __init__(self, data):
        super().__init__(data)
        # t ends up needing to include trial dimension due to condition setup
        self.plot_t = self.t
        self.t = np.tile(self.t, (data["num_trials"], 1))
        self.conditions = data["conditions"]
        self.spikes = data['spikes']
        self.param_names = ["ut", "st", "a_0", "a_1", "a_2", "a_3", "a_4"]
        self.x0 = [500, 100, 0.001, 0.001, 0.001, 0.001, 0.001]
        # mean_delta = 0.10 * (self.window.region_high -
        #                      self.window.region_low)
        # mean_bounds = (
        #     (self.window.region_low - mean_delta),
        #     (self.window.region_high + mean_delta))
        # bounds = (mean_bounds, (0.01, 5.0), (10**-10, 1 / n), (0.001, 1 / n), (0.001, 1 / n), (0.001, 1 / n), (0.001, 1 / n),)
        # self.set_bounds(bounds)

    # def build_function(self, x):
    #     c1 = self.conditions[1]
    #     c2 = self.conditions[2]
    #     c3 = self.conditions[3]
    #     c4 = self.conditions[4]

    #     ut, st, o = x[0], x[1], x[2]
    #     a1, a2, a3, a4 = x[3], x[4], x[5], x[6]

    #     # big_t = (a1 * c1 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
    #     #     a2 * c2 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
    #     #         a3 * c3 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
    #     #             a4 * c4 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))))

    #     fun1 = a1 * np.exp(-np.power(self.t.T - ut, 2.) /
    #                        (2 * np.power(st, 2.)))
    #     fun2 = a2 * np.exp(-np.power(self.t.T - ut, 2.) /
    #                        (2 * np.power(st, 2.)))
    #     fun3 = a3 * np.exp(-np.power(self.t.T - ut, 2.) /
    #                        (2 * np.power(st, 2.)))
    #     fun4 = a4 * np.exp(-np.power(self.t.T - ut, 2.) /
    #                        (2 * np.power(st, 2.)))

    #     inside_sum = (self.spikes.T * (-np.log(o + c1*fun1 + c2*fun2 + c3*fun3 + c4*fun4)) +
    #                   (1 - self.spikes.T) * (-np.log(1 - (o + c1*fun1 + c2*fun2 + c3*fun3 + c4*fun4))))
    #     objective = np.sum(inside_sum)

    #     return objective

    def model(self, x, plot=False, condition=False):
        c1 = self.conditions[1]
        c2 = self.conditions[2]
        c3 = self.conditions[3]
        c4 = self.conditions[4]

        ut, st, a_0 = x[0], x[1], x[2]
        a_1, a_2, a_3, a_4 = x[3], x[4], x[5], x[6]

        fun1 = a_1 * np.exp(-np.power(self.t - ut, 2.) /
                           (2 * np.power(st, 2.)))
        fun2 = a_2 * np.exp(-np.power(self.t - ut, 2.) /
                           (2 * np.power(st, 2.)))
        fun3 = a_3 * np.exp(-np.power(self.t - ut, 2.) /
                           (2 * np.power(st, 2.)))
        fun4 = a_4 * np.exp(-np.power(self.t - ut, 2.) /
                           (2 * np.power(st, 2.)))

        if plot:
            if condition==1:
                return ((a_1  * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.)))))
            elif condition==2:
                return ((a_2  * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.)))))
            elif condition==3:
                return ((a_3  * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.)))))
            elif condition ==4:
                return ((a_4  * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.)))))
            else:
                return (
                    ((a_1  * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.))))) +
                    ((a_2  * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.))))) +
                    ((a_3  * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.))))) + 
                    ((a_4  * np.exp(-np.power(self.plot_t - ut, 2.) / (2 * np.power(st, 2.))))) + a_0
                )

        return (c1*fun1 + c2*fun2 + c3*fun3 + c4*fun4)+ a_0

    def objective(self, x):
        fun = self.model(x)
        return np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

    def update_params(self):

        self.ut = self.fit[0]
        self.st = self.fit[1]
        self.o = self.fit[2]
        self.a1 = self.fit[3]
        self.a2 = self.fit[4]
        self.a3 = self.fit[5]
        self.a4 = self.fit[6]

    def pso_con(self, x):

        return 1 - (x[3] + x[4] + x[2] + x[5] + x[6])

class PlaceField(Model):

    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.param_names = ["ut_x", "st_x","ut_y", "st_y", "a_0", "a_1"]

    def info_callback(self):
        if "pos" in self.info:
            self.posx = self.info["pos"][:,0]
            self.posy = self.info["pos"][:,1]

    def objective(self, x):
        fun = self.model(x)
        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def model(self, x):
        utx, stx, uty, sty, a_0, a_1 = x

        self.function = (
            (a_1 * np.exp(-(np.power(self.posx - utx, 2.) / (2 * np.power(stx, 2.)) +
                np.power(self.posy - uty, 2.) / (2 * np.power(sty, 2.))))) + a_0)
        return self.function

    def plot_model(self, x):
        utx, stx, uty, sty, a_0, a_1 = x

        self.function = (
            (a_1 * np.exp(-(np.power(self.posx - utx, 2.) / (2 * np.power(stx, 2.)) +
                np.power(self.posy - uty, 2.) / (2 * np.power(sty, 2.))))) + a_0)
        return self.function



class ConstCat(Model):

    """Model which contains seperate constant terms per each given category.

    Parameters
    ----------
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.
    a1 : float
        Coefficient of category 1 gaussian distribution.
    a2 : float
        Coefficient of category 2 gaussian distribution.
    a3 : float
        Coefficient of category 3 gaussian distribution.
    a4 : float
        Coefficient of category 4 gaussian distribution.

    """

    def __init__(
            self,
            spikes,
            time_low,
            time_high,
            time_bin,
            bounds,
            conditions):
        super().__init__(spikes, time_low, time_high, time_bin, bounds)
        self.name = "Constant-Category"
        self.conditions = conditions
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.a4 = None

    def build_function(self, x):
        c1 = self.conditions[1]
        c2 = self.conditions[2]
        c3 = self.conditions[3]
        c4 = self.conditions[4]
        a1, a2, a3, a4 = x[0], x[1], x[2], x[3]
        big_t = a1 * c1 + a2 * c2 + a3 * c3 + a4 * c4
        return np.sum(self.spikes.T * (-np.log(big_t)) +
                      (1 - self.spikes.T) * (-np.log(1 - (big_t))))

    def fit_params(self):
        super().fit_params()
        self.a1 = self.fit[0]
        self.a2 = self.fit[1]
        self.a3 = self.fit[2]
        self.a4 = self.fit[3]

        return self.fit, self.fun

    def pso_con(self, x):
        return 1


class PositionTime(Model):

    def __init__(self, data):
        super().__init__(data)
        self.name = "time_position"
        #self.num_params = 8
        self.num_params = 10
        self.ut = None
        self.st = None
        self.a = None
        self.o = None
        n = 4
        mean_delta = 0.10 * (self.window.time_high -
                             self.window.time_low)
        mean_bounds = (
            (self.window.time_low - mean_delta),
            (self.window.time_high + mean_delta))
        bounds = ((0.001, 1 / n), mean_bounds, (0.01, 5.0), (10**-10, 1 / n),
                  (0.001, 1 / n), mean_bounds, (0.01, 5.0), (10**-10, 1 / n),
                  mean_bounds, (0.01, 5.0))

        self.set_bounds(bounds)
        self.position = data["position"]

    def build_function(self, x):
        a_t, mu_t, sig_t, a_t0 = x[0], x[1], x[2], x[3]
        a_s, mu_s, sig_s, a_s0 = x[4], x[5], x[6], x[7]
        mu_sy, sig_sy = x[8], x[9]
        time_comp = (
            (a_t * np.exp(-np.power(self.t - mu_t, 2.) / (2 * np.power(sig_t, 2.)))) + a_t0)

        #spacial_comp = a_s * (np.exp(-np.power(self.position[0] - mu_s, 2.) / (2 * np.power(sig_s, 2.)))) + a_s0

        spacial_comp = a_s * (np.exp(-np.power(self.position[1] - mu_s, 2.) / (2 * np.power(sig_s, 2.))
                                     + (np.power(self.position[0] - mu_sy, 2.) / (2 * np.power(sig_sy, 2.))))) + a_s0

        self.function = time_comp + spacial_comp
        res = np.sum(self.spikes * (-np.log(self.function)) +
                     (1 - self.spikes) * (-np.log(1 - (self.function))))
        return res

    def fit_params(self):
        super().fit_params()
        return (self.fit, self.fun)

    def pso_con(self, x):
        return 1 - (x[0] + x[3] + x[4] + x[7])

    def expose_fit(self):
        if self.fit is None:
            raise ValueError("fit not yet computed")
        else:
            self.a_t = self.fit[0]
            self.mu_t = self.fit[1]
            self.sig_t = self.fit[2]
            self.a_t0 = self.fit[3]
            self.a_s = self.fit[4]
            self.mu_s = self.fit[5]
            self.sig_s = self.fit[6]
            self.a_s0 = self.fit[7]
            self.mu_sy = self.fit[8]
            self.sig_sy = self.fit[9]
        time_comp = (
            (self.a_t * np.exp(-np.power(self.t - self.mu_t, 2.) / (2 * np.power(self.sig_t, 2.)))) + self.a_t0)

        #spacial_comp = self.a_s * (np.exp(-np.power(self.position[0] - self.mu_s, 2.) / (2 * np.power(self.sig_s, 2.)))) + self.a_s0

        spacial_comp = self.a_s * (np.exp(-np.power(self.position[1] - self.mu_s, 2.) / (2 * np.power(self.sig_s, 2.))
                                          + (np.power(self.position[0] - self.mu_sy, 2.) / (2 * np.power(self.sig_sy, 2.))))) + self.a_s0
        fun = time_comp + spacial_comp
        return fun


class PositionGauss(Model):

    def __init__(self, data):
        super().__init__(data)
        self.name = "pos-gauss"
        #self.num_params = 8
        self.num_params = 4
        self.ut = None
        self.st = None
        self.a = None
        self.o = None
        n = 3
        mean_delta = 0.10 * (self.window.time_high -
                             self.window.time_low)
        mean_bounds = (
            (self.window.time_low - mean_delta),
            (self.window.time_high + mean_delta))
        bounds = ((10**-10, 1 / n), (0, 1000),
                  (0.01, 500.0), (10**-10,  1 / n))

        self.set_bounds(bounds)
        self.position = data["position"]

    def build_function(self, x):
        a, mu_x, sig_x, a_0 = x[0], x[1], x[2], x[3]
        xpos = np.arange(0, 995, 1)
        self.function = (
            a * (np.exp(-np.power(xpos - mu_x, 2.) / (2 * np.power(sig_x, 2.))))) + a_0
        res = np.sum(self.spikes.T * (-np.log(self.function)) +
                     (1 - self.spikes.T) * (-np.log(1 - (self.function))))
        return res

    def fit_params(self):
        super().fit_params()
        return (self.fit, self.fun)

    def pso_con(self, x):
        return 1 - (x[0] + x[3])

    def expose_fit(self):
        if self.fit is None:
            raise ValueError("fit not yet computed")
        else:
            self.a = self.fit[0]
            self.mu_x = self.fit[1]
            self.sig_x = self.fit[2]
            self.a_0 = self.fit[3]

        xpos = np.arange(0, 990, 1)

        fun = self.a * (np.exp(-np.power(xpos - self.mu_x, 2.) /
                               (2 * np.power(self.sig_x, 2.)))) + self.a_0
        return fun


class BoxCategories(Model):
    def __init__(self, data):
        super().__init__(data)
        self.name = ("moving_box")
        n = 7
        self.model_type = "position"
        self.spikes = data['spikes_pos']
        self.pos_info = data['pos_info']
        self.num_params = 19
        self.conditions = data["conditions"]
        self.x_pos = np.arange(0, 995, 1)
        self.x_pos = np.tile(self.x_pos, (60, 1)).T
        # bounds = ((10**-10, 1 / n), (0, 1000), (1, 500.0), (10**-10, 1 / n), (0, 1000), (1, 500.0),
        #     (10**-10, 1 / n), (0, 1000), (1, 500.0), (10**-10, 1 / n), (0, 1000), (1, 500.0),
        #     (10**-10, 1 / n), (0, 1000), (1, 500.0), (10**-10, 1 / n), (0, 1000), (1, 500.0),
        #     (10**-10,  1 / n))

        # self.set_bounds(bounds)

    def build_function(self, x):
        a1, mu_x1, sig_x1 = x[0], x[1], x[2]
        a2, mu_x2, sig_x2 = x[3], x[4], x[5]
        a3, mu_x3, sig_x3 = x[6], x[7], x[8]
        a4, mu_x4, sig_x4 = x[9], x[10], x[11]
        a5, mu_x5, sig_x5 = x[12], x[13], x[14]
        a6, mu_x6, sig_x6 = x[15], x[16], x[17]
        a_0 = x[18]
        c1 = self.conditions[1]
        c2 = self.conditions[2]
        c3 = self.conditions[3]
        c4 = self.conditions[4]
        c5 = self.conditions[5]
        c6 = self.conditions[6]

        # xpos = np.arange(0, 995, 1)
        # xpos =  np.tile(xpos, (60, 1)).T

        pos1 = (a1 * c1 * np.exp(-np.power(self.x_pos -
                                           mu_x1, 2.) / (2 * np.power(sig_x1, 2.))))
        pos2 = (a2 * c2 * np.exp(-np.power(self.x_pos -
                                           mu_x2, 2.) / (2 * np.power(sig_x2, 2.))))
        pos3 = (a3 * c3 * np.exp(-np.power(self.x_pos -
                                           mu_x3, 2.) / (2 * np.power(sig_x3, 2.))))
        pos4 = (a4 * c4 * np.exp(-np.power(self.x_pos -
                                           mu_x4, 2.) / (2 * np.power(sig_x4, 2.))))
        pos5 = (a5 * c5 * np.exp(-np.power(self.x_pos -
                                           mu_x5, 2.) / (2 * np.power(sig_x5, 2.))))
        pos6 = (a6 * c6 * np.exp(-np.power(self.x_pos -
                                           mu_x6, 2.) / (2 * np.power(sig_x6, 2.))))

        self.function = pos1 + pos2 + pos3 + pos4 + pos5 + pos6 + a_0
        res = np.sum(self.spikes * (-np.log(self.function.T)) +
                     (1 - self.spikes) * (-np.log(1 - (self.function.T))))
        return res

    def update_params(self):
        self.ut = self.fit[0]
        self.st = self.fit[1]
        self.a = self.fit[2]
        self.o = self.fit[3]

    def fit_params(self):
        super().fit_params()
        return (self.fit, self.fun)

    def pso_con(self, x):
        return 1 - (x[0] + x[3])

    def expose_fit(self, category=0):

        if self.fit is None:
            raise ValueError("fit not yet computed")
        else:
            a1, mu_x1, sig_x1 = self.fit[0], self.fit[1], self.fit[2]
            a2, mu_x2, sig_x2 = self.fit[3], self.fit[4], self.fit[5]
            a3, mu_x3, sig_x3 = self.fit[6], self.fit[7], self.fit[8]
            a4, mu_x4, sig_x4 = self.fit[9], self.fit[10], self.fit[11]
            a5, mu_x5, sig_x5 = self.fit[12], self.fit[13], self.fit[14]
            a6, mu_x6, sig_x6 = self.fit[15], self.fit[16], self.fit[17]
            a_0 = self.fit[18]

            cat_coefs = [a1, a2, a3, a4, a5, a6]
            mu_cat = [mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, mu_x6]
            sig_cat = [sig_x1, sig_x2, sig_x3, sig_x4, sig_x5, sig_x6]

            fun = cat_coefs[category-1] * np.exp(-np.power(
                self.x_pos - mu_cat[category-1], 2.) / (2 * np.power(sig_cat[category-1], 2.)))

            return fun


class TimePos(Model):

    """Model which contains a time dependent gaussian compenent and an offset parameter.

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    num_params : int
        Integer signifying the number of model parameters.
    spikes : dict     
        Dict containing binned spike data for current cell.

    """

    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.position = data['spike_info']['position']
        self.name = "time"
        self.num_params = 5

    def build_function(self, x):
        # pso stores params in vector x
        a, ut, st, o, p = x[0], x[1], x[2], x[3], x[4]

        self.function = (
            (a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o + p*np.array(self.position))
        res = np.sum(self.spikes * (-np.log(self.function)) +
                     (1 - self.spikes) * (-np.log(1 - (self.function))))
        return res

    def fit_params(self):
        super().fit_params()
        return (self.fit, self.fun)

    def pso_con(self, x):
        return 1 - (x[0] + x[3])

    def expose_fit(self):
        if self.fit is None:
            raise ValueError("fit not yet computed")
        else:
            self.a = self.fit[0]
            self.ut = self.fit[1]
            self.st = self.fit[2]
            self.o = self.fit[3]
            self.p = self.fit[4]
        fun = (self.a * np.exp(-np.power(self.t - self.ut, 2.) /
                               (2 * np.power(self.st, 2.)))) + self.o + self.p * (np.sum(self.position, axis=0) / self.num_trials)
        return fun

class TimeVariableLength(Model):

    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.param_names = ["a_1", "ut", "st", "a_0"]
        self.x0 = [1e-5, 100, 100, 1e-5]

    def info_callback(self):
        self.trial_lengths = self.info["trial_length"]
        for ind, trial in enumerate(self.trial_lengths):
            self.spikes[ind][trial:] = np.nan

    def objective(self, x):

        fun = self.model(x)
        total = 0
        for ind, trial in enumerate(self.spikes):
                if self.window[ind, 0] < 0:
                    min_ind = 0
                else:
                    min_ind = self.window[ind, 0]
                
                total+= np.sum(trial[min_ind:self.window[ind, 1]] * (-np.log(fun[min_ind:self.window[ind, 1]])) +
                            (1 - trial[min_ind:self.window[ind, 1]]) * (-np.log(1 - (fun[min_ind:self.window[ind, 1]]))))
        return total

    def model(self, x):
        a, ut, st, o = x

        self.function = (
            (a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        return self.function 
    
    def plot_model(self, x):
        a, ut, st, o = x

        self.function = (
            (a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        return self.function

class SigmaMuTauVariableLength(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu", "tau", "a_1", "a_0"]

    def info_callback(self):
        self.trial_lengths = self.info["trial_length"]
        for ind, trial in enumerate(self.trial_lengths):
            self.spikes[ind][trial:] = np.nan

    def objective(self, x):

        fun = self.model(x)
        total = 0
        for ind, trial in enumerate(self.spikes):
                if self.window[ind, 0] < 0:
                    min_ind = 0
                else:
                    min_ind = self.window[ind, 0]
                
                total+= np.sum(trial[min_ind:self.window[ind, 1]] * (-np.log(fun[min_ind:self.window[ind, 1]])) +
                            (1 - trial[min_ind:self.window[ind, 1]]) * (-np.log(1 - (fun[min_ind:self.window[ind, 1]]))))
        return total

    def model(self, x):

        s, mu, tau, a_1, a_0 = x
        l = 1/tau
        '''old method'''
        # fun = a_1*np.exp(-0.5*(np.power((self.t-m)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-m)/s))))
        # ) + a_0

        self.function = (a_1*(np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))) + a_0

        return self.function
    
    def plot_model(self, x):
        return self.model(x)

class DualVariableLength(Model):

    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.param_names = ["a_1", "a_2", "ut_1", "ut_2", "st_1", "st_2", "a_0"]
        # self.x0 = [1e-5, 1e-5, 100, 100, 1e-5]

    def info_callback(self):
        self.trial_lengths = self.info["trial_length"]
        for ind, trial in enumerate(self.trial_lengths):
            self.spikes[ind][trial:] = np.nan

    def objective(self, x):

        fun = self.model(x)
        total = 0
        # for ind, trial in enumerate(self.spikes):
        #     total+= np.sum(trial[:self.window[ind]] * (-np.log(fun[:self.window[ind]])) +
        #               (1 - trial[:self.trial_lengths[ind]]) * (-np.log(1 - (fun[:self.trial_lengths[ind]]))))
        #         total = 0
        for ind, trial in enumerate(self.spikes):
                if self.window[ind, 0] < 0:
                    min_ind = 0
                else:
                    min_ind = self.window[ind, 0]
                
                total+= np.sum(trial[min_ind:self.window[ind, 1]] * (-np.log(fun[min_ind:self.window[ind, 1]])) +
                            (1 - trial[min_ind:self.window[ind, 1]]) * (-np.log(1 - (fun[min_ind:self.window[ind, 1]]))))
        # l = lambda x: np.sum(self.spikes[x[0]][:self.trial_lengths[x[0]]] * (-np.log(fun[:self.trial_lengths[x[0]]])) +
        #               (1 - self.spikes[x[0]][:self.trial_lengths[x[0]]]) * (-np.log(1 - (fun[:self.trial_lengths[x[0]]]))))
        # obj = map(l, enumerate(self.trial_lengths))
        # obj = np.sum(np.array(list(mapper)) * (-np.log(np.array(list(mapper2)))) +
        #               (1 - np.array(list(mapper))) * (-np.log(1 - (np.array(list(mapper2))))))
        # obj = total
        return total
        # return total

    def model(self, x):
        a_1, a_2, ut_1, ut_2, st_1, st_2, o = x

        self.function = (
            (a_1 * np.exp(-np.power(self.t - ut_1, 2.) / (2 * np.power(st_1, 2.)))) + 
            (a_2 * np.exp(-np.power(self.t - ut_2, 2.) / (2 * np.power(st_2, 2.)))) + o)

        return self.function 
    
    def plot_model(self, x):
       return self.model(x)

class AbsPosVariable(Model):

    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.param_names = ["a_1", "ut", "st", "a_0"]
        # self.x0 = [1e-5, 100, 100, 1e-5]

    def info_callback(self):
        # if "trial_length" in self.info:
        #     self.trial_lengths = self.info["trial_length"]
        #     for ind, trial in enumerate(self.trial_lengths):
        #         self.spikes[ind][trial:] = 0
        #     self.info.pop("trial_length")
        
        if "abs_pos" in self.info:
            pos = self.info["abs_pos"]
            longest_trial = max(list(map(lambda x: len(x), pos)))
            self.pos2 = np.zeros((len(pos), longest_trial),dtype=float)
            for trial in range(len(pos)):
                self.pos2[trial][:len(pos[trial])] = (np.array(pos[trial], dtype=float))
            self.info.pop("abs_pos")

    def objective(self, x):

        fun = self.model(x)
        total = 0
        for ind, trial in enumerate(self.spikes):
                total+= np.sum(trial[self.window[ind, 0]:self.window[ind, 1]] * (-np.log(fun[ind,self.window[ind, 0]:self.window[ind, 1]])) +
                            (1 - trial[self.window[ind, 0]:self.window[ind, 1]]) * (-np.log(1 - (fun[ind,self.window[ind, 0]:self.window[ind, 1]]))))

        return total

    def model(self, x):
        a_1, ut, st, o = x

        self.function = (
            (a_1 * np.exp(-np.power(self.pos2 - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        return self.function

    def plot_model(self, x):
        a, ut, st, o = x

        return ((a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)

class AbsPosVelocity(Model):

    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.param_names = ["a_v", "ut", "st", "a_0"]
        # self.x0 = [1e-5, 100, 100, 1e-5]

    def info_callback(self):
        # if "trial_length" in self.info:
        #     self.trial_lengths = self.info["trial_length"]
        #     for ind, trial in enumerate(self.trial_lengths):
        #         self.spikes[ind][trial:] = 0
        #     self.info.pop("trial_length")
        
        if "abs_pos" in self.info:
            pos = self.info["abs_pos"]
            longest_trial = max(list(map(lambda x: len(x), pos)))
            self.pos2 = np.zeros((pos.shape[0], longest_trial),dtype=float)
            for trial in range(len(pos)):
                self.pos2[trial][:len(pos[trial])] = (np.array(pos[trial], dtype=float))
            self.info.pop("abs_pos")

        if "velocity" in self.info:
            velocity = self.info["velocity"]
            v_max = 0
            for trial in self.info["velocity"]:
                new = max(trial)
                if new > v_max:
                    v_max = new
            longest_trial = max(list(map(lambda x: len(x), velocity)))   
            self.velocity = np.zeros((velocity.shape[0], longest_trial), dtype=float)
            for trial in range(len(velocity)):
                self.velocity[trial][:len(velocity[trial])] = np.array(velocity[trial], dtype=float)
            # velocity = np.array([np.array(xi) for xi in self.info["velocity"]])
            self.velocity = self.velocity/v_max
            self.info.pop("velocity")

    def objective(self, x):

        fun = self.model(x)
        total = 0
        for ind, trial in enumerate(self.spikes):
                total+= np.sum(trial[self.window[ind, 0]:self.window[ind, 1]] * (-np.log(fun[ind,self.window[ind, 0]:self.window[ind, 1]])) +
                            (1 - trial[self.window[ind, 0]:self.window[ind, 1]]) * (-np.log(1 - (fun[ind,self.window[ind, 0]:self.window[ind, 1]]))))

        return total

    def model(self, x):
        a_v, ut, st, o = x

        self.function = (
            (a_v*self.velocity * np.exp(-np.power(self.pos2 - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        return self.function

    def plot_model(self, x):
        a, ut, st, o = x

        return ((a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
     
class RelPosVelocity(Model):

    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.param_names = ["a_v", "ut", "st", "a_0"]
        # self.x0 = [1e-5, 100, 100, 1e-5]

    def info_callback(self):
        # if "trial_length" in self.info:
        #     self.trial_lengths = self.info["trial_length"]
        #     for ind, trial in enumerate(self.trial_lengths):
        #         self.spikes[ind][trial:] = np.nan
        #     self.info.pop("trial_length")

        if "rel_pos" in self.info:
            pos = self.info["rel_pos"]
            longest_trial = max(list(map(lambda x: len(x), pos)))
            self.pos2 = np.zeros((pos.shape[0],longest_trial), dtype=float)
            for trial in range(len(pos)):
                self.pos2[trial][:len(pos[trial])] = (np.array(pos[trial], dtype=float))
            self.info.pop("rel_pos")

        if "velocity" in self.info:
            velocity = self.info["velocity"]
            v_max = 0
            for trial in self.info["velocity"]:
                new = max(trial)
                if new > v_max:
                    v_max = new
            # for uneven sheehan data
            longest_trial = max(list(map(lambda x: len(x), velocity)))   
            if self.pos2.shape[1] < longest_trial:
                self.velocity = np.zeros((velocity.shape[0], self.pos2.shape[1]), dtype=float)
            elif self.pos2.shape[1] <= longest_trial:
                self.velocity = np.zeros((velocity.shape[0], longest_trial), dtype=float)
            for trial in range(len(velocity)):
                # if len(velocity[trial]) > self.velocity.shape[1]:
                #     insert = np.array(velocity[trial], dtype=float)[:self.velocity.shape[1]]
                # else:
                insert = np.array(velocity[trial], dtype=float)
                self.velocity[trial][:len(insert)] = insert
            # for trial in range(len(velocity)):
            #     insert = np.array(velocity[trial], dtype=float)
            #     self.velocity[trial][:len(insert)] = insert
            velocity = np.array([np.array(xi) for xi in self.info["velocity"]])
            self.velocity = self.velocity/v_max
            self.info.pop("velocity")

    def objective(self, x):

        fun = self.model(x)
        total = 0
        # for ind, trial in enumerate(self.spikes):
        #         total+= np.sum(trial[self.window[0][ind]:self.window[1][ind]] * (-np.log(fun[ind,self.window[0][ind]:self.window[1][ind]])) +
        #                     (1 - trial[self.window[0][ind]:self.window[1][ind]]) * (-np.log(1 - (fun[ind,self.window[0][ind]:self.window[1][ind]]))))
        for ind, trial in enumerate(self.spikes):
            total+= np.sum(trial[self.window[ind, 0]:self.window[ind, 1]] * (-np.log(fun[ind,self.window[ind, 0]:self.window[ind, 1]])) +
                        (1 - trial[self.window[ind, 0]:self.window[ind, 1]]) * (-np.log(1 - (fun[ind,self.window[ind, 0]:self.window[ind, 1]]))))

        return total

    def model(self, x):
        a_v, ut, st, o = x

        # self.pos = np.array(list(map(lambda x: np.array(x), self.info["abs_pos"][11])),dtype=float)
        # self.function = np.array((map(lambda x: (a * np.exp(-np.power(self.pos2[x, :self.trial_lengths[x]] - ut, 2.) / (2 * np.power(st, 2.)))) + o, range(self.num_trials))))
        self.function = (
            (a_v * self.velocity * np.exp(-np.power(self.pos2 - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        return self.function

    def plot_model(self, x):
        a, ut, st, o = x
        return ((a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)


class RelPosVariable(Model):

    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.param_names = ["a_1", "ut", "st", "a_0"]
        # self.x0 = [1e-5, 100, 100, 1e-5]

    def info_callback(self):
        # if "trial_length" in self.info:
        #     self.trial_lengths = self.info["trial_length"]
        #     for ind, trial in enumerate(self.trial_lengths):
        #         self.spikes[ind][trial:] = np.nan
        #     self.info.pop("trial_length")

        if "rel_pos" in self.info:
            pos = self.info["rel_pos"]
            longest_trial = max(list(map(lambda x: len(x), pos)))
            self.pos2 = np.zeros((len(pos),longest_trial), dtype=float)
            for trial in range(len(pos)):
                self.pos2[trial][:len(pos[trial])] = (np.array(pos[trial], dtype=float))
            self.info.pop("rel_pos")

    def objective(self, x):

        fun = self.model(x)
        total = 0
        # for ind, trial in enumerate(self.spikes):
        #         total+= np.sum(trial[self.window[0][ind]:self.window[1][ind]] * (-np.log(fun[ind,self.window[0][ind]:self.window[1][ind]])) +
        #                     (1 - trial[self.window[0][ind]:self.window[1][ind]]) * (-np.log(1 - (fun[ind,self.window[0][ind]:self.window[1][ind]]))))
        for ind, trial in enumerate(self.spikes):
            total+= np.sum(trial[self.window[ind, 0]:self.window[ind, 1]] * (-np.log(fun[ind,self.window[ind, 0]:self.window[ind, 1]])) +
                        (1 - trial[self.window[ind, 0]:self.window[ind, 1]]) * (-np.log(1 - (fun[ind,self.window[ind, 0]:self.window[ind, 1]]))))

        return total

    def model(self, x):
        a_1, ut, st, o = x

        # self.pos = np.array(list(map(lambda x: np.array(x), self.info["abs_pos"][11])),dtype=float)
        # self.function = np.array((map(lambda x: (a * np.exp(-np.power(self.pos2[x, :self.trial_lengths[x]] - ut, 2.) / (2 * np.power(st, 2.)))) + o, range(self.num_trials))))
        self.function = (
            (a_1 * np.exp(-np.power(self.pos2 - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        return self.function

    def plot_model(self, x):
        a, ut, st, o = x
        return ((a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)


class ConstVariable(Model):

    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.param_names = ["a_0"]
        # self.x0 = [0.1]

    # def info_callback(self):
        # if "trial_length" in self.info:
        #     self.trial_lengths = self.info["trial_length"]
        #     for ind, trial in enumerate(self.trial_lengths):
        #         self.spikes[ind][trial:] = np.nan
        #     self.info.pop("trial_length")

    def model(self, x, plot=False):
        o = x
        return o

    def objective(self, x):

        fun = self.model(x)
        # if True:
        #     import numpy.ma as ma
        #     spikes = ma.masked_array(self.spikes, self.filter_spikes())
        #     return  np.sum(spikes * (-np.log(fun)) +
        #               (1 - spikes) * (-np.log(1 - (fun))))
        # test = {}
        # for ind, trial in enumerate(self.spikes):
        #     test[ind] = np.array(trial[self.window[ind,0]:self.window[ind, 1]])
        # return (np.sum(test) * (-np.log(fun)) +
        #               (1 - test) * (-np.log(1 - (fun)))))


        total = 0
        for ind, trial in enumerate(self.spikes):
                total+= np.sum(trial[self.window[ind,0]:self.window[ind, 1]] * (-np.log(fun)) +
                            (1 - trial[self.window[ind,0]:self.window[ind, 1]]) * (-np.log(1 - (fun))))

        return total

    def plot_model(self, x):

        o = x 
        return o

    def subselect_spikes(self, spikes):
        return self.spikes[ind][self.window[ind,0]:self.window[ind, 1]]

    def filter_spikes(self):
        mask = np.zeros(self.spikes.shape)

        
        for ind, trial in enumerate(self.spikes):
            
            mask[ind][0:self.window[ind,0]] = 1
            mask[ind][self.window[ind,1]:] = 1

        return masK

class DualPeakedRel(Model):
    
    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.param_names = [
            "ut_a", 
            "st_a", 
            "a_0a",
            "a_1a",  
            "ut_b", 
            "st_b", 
            "a_0b",
            "a_1b"
        ]

    def info_callback(self):
    # if "trial_length" in self.info:
    #     self.trial_lengths = self.info["trial_length"]
    #     for ind, trial in enumerate(self.trial_lengths):
    #         self.spikes[ind][trial:] = np.nan
    #     self.info.pop("trial_length")

        if "rel_pos" in self.info:
            pos = self.info["rel_pos"]
            longest_trial = max(list(map(lambda x: len(x), pos)))
            self.pos2 = np.zeros((len(pos),longest_trial), dtype=float)
            for trial in range(len(pos)):
                self.pos2[trial][:len(pos[trial])] = (np.array(pos[trial], dtype=float))
            self.info.pop("rel_pos")
            
    def objective(self, x):
        fun = self.model(x)
        total = 0
        for ind, trial in enumerate(self.spikes):
            total+= np.sum(trial[self.window[ind, 0]:self.window[ind, 1]] * (-np.log(fun[ind,self.window[ind, 0]:self.window[ind, 1]])) +
                        (1 - trial[self.window[ind, 0]:self.window[ind, 1]]) * (-np.log(1 - (fun[ind,self.window[ind, 0]:self.window[ind, 1]]))))

        return total


    def model(self, x):
        ut_a, st_a, a_0a, a_1a, ut_b, st_b, a_0b, a_1b = x

        fun1 = (
            (a_1a * np.exp(-np.power(self.pos2 - ut_a, 2.) / (2 * np.power(st_a, 2.)))) + a_0a)
        fun2 = (
            (a_1b * np.exp(-np.power(self.pos2 - ut_b, 2.) / (2 * np.power(st_b, 2.)))) + a_0b)
        self.function = fun1 + fun2
        
        return self.function

    def plot_model(self, x):
        ut_a, st_a, a_0a, a_1a, ut_b, st_b, a_0b, a_1b = x

        fun1 = (
            (a_1a * np.exp(-np.power(self.t - ut_a, 2.) / (2 * np.power(st_a, 2.)))) + a_0a)
        fun2 = (
            (a_1b * np.exp(-np.power(self.t - ut_b, 2.) / (2 * np.power(st_b, 2.)))) + a_0b)
        self.function = fun1 + fun2
        
        return self.function

class DualPeakedAbs(Model):
    
    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.param_names = [
            "ut_a", 
            "st_a", 
            "a_0a",
            "a_1a",  
            "ut_b", 
            "st_b", 
            "a_0b",
            "a_1b"
        ]

    def info_callback(self):
    # if "trial_length" in self.info:
    #     self.trial_lengths = self.info["trial_length"]
    #     for ind, trial in enumerate(self.trial_lengths):
    #         self.spikes[ind][trial:] = np.nan
    #     self.info.pop("trial_length")

        if "abs_pos" in self.info:
            pos = self.info["abs_pos"]
            longest_trial = max(list(map(lambda x: len(x), pos)))
            self.pos2 = np.zeros((len(pos),longest_trial), dtype=float)
            for trial in range(len(pos)):
                self.pos2[trial][:len(pos[trial])] = (np.array(pos[trial], dtype=float))
            self.info.pop("abs_pos")
            
    def objective(self, x):
        fun = self.model(x)
        total = 0
        for ind, trial in enumerate(self.spikes):
            total+= np.sum(trial[self.window[ind, 0]:self.window[ind, 1]] * (-np.log(fun[ind,self.window[ind, 0]:self.window[ind, 1]])) +
                        (1 - trial[self.window[ind, 0]:self.window[ind, 1]]) * (-np.log(1 - (fun[ind,self.window[ind, 0]:self.window[ind, 1]]))))

        return total


    def model(self, x):
        ut_a, st_a, a_0a, a_1a, ut_b, st_b, a_0b, a_1b = x

        fun1 = (
            (a_1a * np.exp(-np.power(self.pos2 - ut_a, 2.) / (2 * np.power(st_a, 2.)))) + a_0a)
        fun2 = (
            (a_1b * np.exp(-np.power(self.pos2 - ut_b, 2.) / (2 * np.power(st_b, 2.)))) + a_0b)
        self.function = fun1 + fun2
        
        return self.function

    def plot_model(self, x):
        ut_a, st_a, a_0a, a_1a, ut_b, st_b, a_0b, a_1b = x

        fun1 = (
            (a_1a * np.exp(-np.power(self.t - ut_a, 2.) / (2 * np.power(st_a, 2.)))) + a_0a)
        fun2 = (
            (a_1b * np.exp(-np.power(self.t - ut_b, 2.) / (2 * np.power(st_b, 2.)))) + a_0b)
        self.function = fun1 + fun2
        
        return self.function




