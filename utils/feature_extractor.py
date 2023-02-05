class FeatureExtractor() :
    features =  None
    def __init__(self, m) :
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.detach().cpu()
        self.features_before = input[0].detach().cpu()