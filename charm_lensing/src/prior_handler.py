#!/usr/bin/env python3
import nifty8 as ift
from charm_lensing.src import utils
from functools import reduce, partial


class ParametricPriorHandler:
    def __init__(self, prefix, instantiation_dict):
        self.constants = {}
        self.operators = []

        for key, val in instantiation_dict.items():
            domain_key = '_'.join((prefix, key))
            operator = self.get_operator(domain_key, val)

            if isinstance(operator, dict):
                self.constants.update(operator)
            else:
                self.operators.append(operator)

        self.constant_parameter_operator = partial(utils.unite_dict, b=self.constants)
        self.free_parameter_operator = reduce(lambda x, y: x + y, self.operators)

    @staticmethod
    def get_operator(domain_key, values):
        values['key'] = domain_key

        distribution = values.pop('distribution')

        if distribution in ['uniform']:
            operator = ift.UniformOperator(
                ift.UnstructuredDomain(values['N_copies']),
                loc=values['mean'],
                scale=values['sigma']
            ).ducktape(domain_key).ducktape_left(values['key'])

        elif distribution in ['normal']:
            operator = ift.NormalTransform(**values).ducktape_left(domain_key)

        elif distribution in ['log_normal', 'lognormal']:
            operator = ift.LognormalTransform(**values).ducktape_left(domain_key)

        elif distribution is None:
            value = ift.Field.from_raw(
                ift.UnstructuredDomain(values['N_copies']), values['mean']
            )
            print(f'Constant ({values["key"]}): {value.val}')
            operator = {domain_key: value.val}

        else:
            print('This distribution is not implemented')
            raise NotImplementedError

        return operator
