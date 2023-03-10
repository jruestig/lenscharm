#!/usr/bin/env python3
import nifty8 as ift
from charm_lensing.src import utils
from functools import reduce, partial


class ParamatricPrior:
    def __init__(self, prefix, instantiation_dictionary):

        constants = {}
        operators = []

        for key, val in instantiation_dictionary.items():
            domain_key = '_'.join((prefix, key))
            operator = self.distribution_getter(domain_key, val)

            if type(operator) is dict:
                constants = utils.unite_dict(constants, operator)

            else:
                operators.append(operator)

        self.constant_parameter_operator = partial(utils.unite_dict, b=constants)
        self.free_parameter_operator = reduce(lambda x,y: x+y, operators)


    @staticmethod
    def distribution_getter(domain_key: str, values: dict):
        values['key'] = domain_key

        distribution = values.pop('distribution')
        if distribution in ['uniform']:
            return ift.UniformOperator(
                ift.UnstructuredDomain(values['N_copies']),
                loc=values['mean'],
                scale=values['sigma']
            ).ducktape(values['key']).ducktape_left(values['key'])

        elif distribution in ['normal']:
            return ift.NormalTransform(**values).ducktape_left(values['key'])

        elif distribution in ['log_normal', 'lognormal']:
            return ift.LognormalTransform(**values).ducktape_left(values['key'])

        elif distribution is None:
            value = ift.Field.from_raw(
                ift.UnstructuredDomain(values['N_copies']), values['mean']
            )
            print(f'Constant ({values["key"]}): {value.val}')
            return {values['key']: value.val}

        else:
            print('This distribution is not implemented')
            raise NotImplementedError
