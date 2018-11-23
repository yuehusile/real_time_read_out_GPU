"""
==================================
YAML (:mod:`fklab.utilities.yaml`)
==================================

.. currentmodule:: fklab.utilities.yaml

This module will import the complete namespace of the PyYaml package and
add support for OrderedDicts.
    
"""

from __future__ import absolute_import

from yaml import *

from collections import OrderedDict as _OrderedDict

# note that we will use the default mapping tag,
# which means that all maps are loaded as OrderedDicts
_mapping_tag = resolver.BaseResolver.DEFAULT_MAPPING_TAG

def dict_representer(dumper, data):
    return dumper.represent_mapping( _mapping_tag, data.iteritems())

def dict_constructor(loader, node):
    return _OrderedDict( loader.construct_pairs(node))

add_representer( _OrderedDict, dict_representer )
add_constructor( _mapping_tag, dict_constructor )

