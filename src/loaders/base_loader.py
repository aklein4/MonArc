

class BaseLoader:

    def __init__(self, url, train, debug=False):
        self.url = url
        self.name = f'{url.replace("/", "-")}-{"train" if train else "val"}'

        self.train = train
        self.debug = debug


    def reset(self):
        raise NotImplementedError
    

    def __len__(self):
        raise NotImplementedError
    

    def __call__(self, batchsize):
        raise NotImplementedError