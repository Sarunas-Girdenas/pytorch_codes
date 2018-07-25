from numpy.random import uniform, randint

class randomSampler(object):
    """Purpose: uniform samplers
    for floats and integers

    """

    def __init__(self):
        """Constructor,
        might be used later
        """
        return None

    @staticmethod
    def sampleInteger(min_, max_):
        """Purpose: uniformly sample integers
        from the given range
        """

        if min_ >= max_:
            raise ValueError('min value is larger than max value!')

        return randint(min_, max_)

    @staticmethod
    def sampleFloat(min_, max_):
        """Purpose: uniformly sample floats
        from the given range
        """

        if min_ >= max_:
            raise ValueError('min value is larger than max value!')

        return round(uniform(min_, max_, 1)[0], 2)