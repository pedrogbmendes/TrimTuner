from robo.models.fabolas_gp import FabolasGPMCMC


###############################################################
#   Gaussian Processes as implemented in FABOLAS
###############################################################
class EnsembleGPs(FabolasGPMCMC):
    def __init__(self, kernel, basis_func,
                 prior=None, n_hypers=20,
                 chain_length=2000, burnin_steps=2000,
                 normalize_output=False,
                 rng=None,
                 lower=None,
                 upper=None,
                 noise=-8):


        super(EnsembleGPs, self).__init__(kernel, basis_func, prior,
                                            n_hypers, chain_length,
                                            burnin_steps,
                                            normalize_output=normalize_output,
                                            normalize_input=False,
                                            rng=rng, lower=lower,
                                            upper=upper, noise=noise)    