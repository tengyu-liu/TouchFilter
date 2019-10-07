class Model:
    def __init__(self, config):
        self.build_config(config)
        self.build_input()
        self.build_model()
        self.build_train()
        self.build_summary()
    
    def build_config(self, config):
        self.n_pts = 20000
        self.situation_invariant = config.situation_invariant
        self.penalty_strength = config.penalty_strength
    
    def build_input(self):
        pass

    def build_model(self):
        pass
    
    def build_train(self):
        pass
    
    def build_summary(self):
        pass