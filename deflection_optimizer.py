import cluster_fits as cf
import nifty8 as ift
import numpy as np

from os.path import join
from source_fwd import load_fits
from functools import partial


source = load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/tmp_source.fits')
d = load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/glamer_ls_{}arcsec.fits'.format(lensresolution))
deflection_ = np.array((load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/alphax_clus.fits'),
                        load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/alphay_clus.fits')))
deflection_ *= cf.toarcsec

lensresolution = 0.01

d *= 1e9
detectorspace = cf.Space(d.shape, lensresolution)
mask = d > 0

clusternumber = 1
convergence, distance, (M200, R200), subs, weights = cf.get_cluster(
    clusternumber,
    {'simulation': 'bacco',
     'distance': 1.3008352168393837,
     'mapname': 'big_clusters_adj_',
     'path': '/home/jruestig/Data/angulo/tcm_clusters/',
     'redshift': 0.4},
    {'zlens': 0.4, 'zsource': 9.0})
ER = cf.einsteinradius(M200, 0.4, 9.0)
models, (xy_subs, subs_mas, circulars) = cf.model_creator(
    clusternumber,
    {'adjust': True,
     'bacco': '/home/jruestig/pro/python/cluster_fits/output/masked/',
     'barahir': '/home/jruestig/pro/python/cluster_fits/output/barahir/',
     'mass': {'upcut': ['max'], 'locut': [5]},
     'regions': {'box': False, 'inner': ['weights'], 'outer': [1.5]},
     'models': {'gals': False,
                'halocomponents': [2],
                'elliptical': ['dPIESH'],
                'circular': ['fCNFW'],
                'shear': ['SH']}},
    ((ER, R200), distance, convergence.shape),
    subs,
    (weights, 0.05),
    subsgetter=True
)
deflection = cf.load_deflection(
    clusternumber,
    None,
    1.3008352168393837,
    {'simulation': 'bacco',
     'distance': 1.3008352168393837,
     'mapname': 'big_clusters_adj_',
     'path': '/home/jruestig/Data/angulo/tcm_clusters/',
     'redshift': 0.4})
recname, model, (boxr, boxo) = models[0]
recposition = np.load(
    join('/home/jruestig/pro/python/lensing/output/interbeginning', recname+'.npy'),
    allow_pickle=True
).item()
position = np.load(
    join('/home/jruestig/pro/python/lensing/output/interbeginning', recname+'_priorposition.npy'),
    allow_pickle=True
).item()

from NiftyOperators import PriorTransform
prior = PriorTransform(model.get_priorparams())
ddatar = ift.makeField(
    ift.UnstructuredDomain((2,) + (mask.sum().item(),)),
    deflection_[:, mask]
)
dmodel = ift.JaxOperator(
    prior.domain,
    ddatar.domain,
    partial(model.deflection_point, detectorspace.xycoords[:, mask])
    # lambda x: model.deflection_point(detectorspace.xycoords[:, mask], x)
)
egd = ift.GaussianEnergy(ddatar) @ dmodel @ prior
posterior = ift.EnergyAdapter(
    ift.makeField(prior.domain, position),
    ift.StandardHamiltonian(egd),
    want_metric=True)
mini = ift.NewtonCG(ift.AbsDeltaEnergyController(
    1E-8,
    iteration_limit=50,
    name="deflection"))
posterior, _ = mini(posterior)

position = posterior.position
cf.save_as_csv(model.models(),
               prior(position).val,
               prior_position=position.val,
               output=('/home/jruestig/pro/python/lensing/output/interbeginning', recname),
               )
