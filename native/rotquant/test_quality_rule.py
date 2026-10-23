#!/usr/bin/env python3
"""Tests for the frozen quality rule, never a substitute for model evaluation."""
import copy
import unittest
from run_model_qualification import quality


def row(nll=1.0, margin=1.0, agree=True):
    return {'reference_nll':1.0,'nll':nll,'reference_margin':margin,
            'argmax':0 if agree else 1,'reference_argmax':0,
            'logit_cosine':0.99,'kl_reference_candidate':0.01}


class QualityRule(unittest.TestCase):
    def test_primary_bound_is_not_relaxed(self):
        self.assertTrue(quality([row(1.07)])['pass'])
        self.assertFalse(quality([row(1.070001)])['pass'])

    def test_confident_boundary(self):
        rows=[row() for _ in range(1000)]
        rows[0]['argmax']=rows[1]['argmax']=1
        self.assertTrue(quality(rows)['pass'])
        rows[2]['argmax']=1
        self.assertFalse(quality(rows)['pass'])

    def test_ordinary_margin_boundary(self):
        rows=[row(margin=0.1) for _ in range(999)]+[row()]
        for r in rows[:20]:r['argmax']=1
        self.assertTrue(quality(rows)['pass'])
        rows[20]['argmax']=1
        self.assertFalse(quality(rows)['pass'])

    def test_empty_and_missing_denominator_refused(self):
        with self.assertRaises(RuntimeError):quality([])
        with self.assertRaises(RuntimeError):quality([row(margin=0.01)])
        invalid=row();invalid['reference_nll']=0
        with self.assertRaises(RuntimeError):quality([invalid])

    def test_counts_cover_the_rows_actually_supplied(self):
        rows=[row(),row(margin=0.5),row(margin=0.01)]
        result=quality(rows)
        self.assertEqual(result['positions'],3)
        self.assertEqual(result['agreement']['0.1']['positions'],2)
        self.assertEqual(result['agreement']['1.0']['positions'],1)


if __name__=='__main__':
    unittest.main()
