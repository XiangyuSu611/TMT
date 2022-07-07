import logging


logger = logging.getLogger(__name__)


def lazyprop(fn):
    attr_name = '_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            logger.debug("Evaluating lazy property \'{}\'".format(fn.__name__))
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop
