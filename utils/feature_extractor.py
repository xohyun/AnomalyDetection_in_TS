class FeatureExtractor() :
    '''
    Feature extractor
    use in the trainer file
    '''
    features =  None
    def __init__(self, m) :
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        output = output[-1] #################################### 원래 tuple이 아닌디..?
        self.features = output.detach().cpu()
        self.features_before = input[0].detach().cpu()