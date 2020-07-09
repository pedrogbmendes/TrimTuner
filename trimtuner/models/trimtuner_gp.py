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


        super(EnsembleGPs, self).__init__(kernel,
                                            prior=prior,
                                            burnin_steps=burnin_steps,
                                            chain_length=chain_length,
                                            n_hypers=n_hypers,
                                            normalize_output=False,
                                            basis_func=basis_func,
                                            lower=lower,
                                            upper=upper,
                                            rng=rng)
        