import numpy as np
import torch
from models.endpoint_survival_v1.common import biased_state, criticality

def test_criticality_is_predicate_balanced():
    rel=np.array([[0,1,0],[0,2,1],[2,1,1]])
    out=criticality(rel,3,np.array([1,2]))
    assert np.allclose(out,[1.5,1.5,1.0])

def test_semantic_margin_can_make_query_eligible():
    cls=np.array([[0.0,0.0,0.0],[0.0,0.0,0.0]],np.float32); masks=np.ones((2,2,2),np.float32)
    s=biased_state(cls,masks,(2,2),semantic_bias=np.array([1.,0.]),threshold=.5)
    assert s['candidates']

def test_competition_margin_preserves_shape():
    cls=np.array([[3.,0.,-1.],[2.,0.,-1.]],np.float32); masks=np.array([np.ones((2,2)),np.ones((2,2))*1.1],np.float32)
    s=biased_state(cls,masks,(2,2),competition_bias=np.array([0.,1.]))
    assert s['winner_map'].shape==(2,2)
