
class InfiniteLoader:
    """
    A wrapper to make a DataLoader infinite.
    It automatically resets the loader when exhaustion occurs, 
    and increments the epoch for the sampler if applicable.
    """
    def __init__(self, loader, start_epoch=0):
        self.loader = loader
        self.epoch = start_epoch
        self._reset_iterator(start_epoch)
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        #self._reset_iterator(epoch)
    
    def _reset_iterator(self, epoch):
        if hasattr(self.loader.sampler, 'set_epoch'):
            self.loader.sampler.set_epoch(epoch)
        self.iterator = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self._reset_iterator(self.epoch)
            return next(self.iterator)
