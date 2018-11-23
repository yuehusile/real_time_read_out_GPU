# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:14:17 2013

@author: fabian
"""

import nose

import numpy as np

from fklab.segments import Segment, SegmentError

class TestSegments:
    def setUp(self):
        self.s1 = [[0,10],[20,30],[50,100]]
        self.s2 = [[4,6],[15,25],[40,200]]
        self.s3 = [[8,20],[35,60],[150,180]]        
        

def test_segment_construct_from_list():
    s = Segment( [1,2])
    nose.tools.assert_is_instance(s,Segment)

def test_segment_construct_from_nested_list():
    s = Segment( [[1,2],[3,4],[5,6]])
    nose.tools.assert_is_instance(s,Segment)

def test_segment_construct_from_array():
    s = Segment( np.ones((10,2)) )
    nose.tools.assert_is_instance(s,Segment)

def test_segment_construct_empty():
    s = Segment([])
    nose.tools.assert_is_instance(s,Segment)

def test_segment_equal():
    s1 = Segment([[1,2],[3,4],[10,20]])
    s2 = Segment([[1,2],[3,4],[10,20]])
    nose.tools.assert_equal(s1,s2)

@nose.tools.raises(SegmentError)
def test_segment_equal_2():
    Segment( [[1,2],[10,20]] ) == 'a'

def test_segment_exclusive():
    s1 = Segment([[0,10],[20,30],[50,100]])
    s2 = Segment([[4,6],[15,25],[40,200]])
    s1.exclusive(s2)
    np.testing.assert_equal(s1._data, np.array([[0,4],[6,10],[25,30]]))

def test_segment_exclusive_2():
    s1 = Segment([[0,10],[20,30],[50,100]])
    s2 = Segment([[4,6],[15,25],[40,200]])
    s3 = Segment([[8,20],[35,60],[150,180]])
    nose.tools.assert_equal( s1 & ~s2 & ~s3, s1.exclusive(s2,s3) )

def test_segment_union():
    s1 = Segment([[0,10],[20,30],[50,100]])
    s2 = Segment([[4,6],[15,25],[40,200]])
    s1.union(s2)
    np.testing.assert_equal(s1._data, np.array([[0,10],[15,30],[40,200]]))    

def test_segment_union_2():
    s1 = Segment([[0,10],[20,30],[50,100]])
    s2 = Segment([[4,6],[15,25],[40,200]])
    s3 = Segment([[8,20],[35,60],[150,180]])
    nose.tools.assert_equal( s1 | s2 | s3, s1.union(s2,s3) )
    
def test_segment_intersection():
    s1 = Segment([[0,10],[20,30],[50,100]])
    s2 = Segment([[4,6],[15,25],[40,200]])
    s1.intersection(s2)
    np.testing.assert_equal(s1._data, np.array([[4,6],[20,25],[50,100]]))

def test_segment_intersection_2():
    s1 = Segment([[0,10],[20,30],[50,100]])
    s2 = Segment([[4,6],[15,25],[40,200]])
    s3 = Segment([[8,20],[35,60],[150,180]])
    nose.tools.assert_equal( s1 & s2 & s3, s1.intersection(s2,s3) )

def test_segment_difference():
    s1 = Segment([[0,10],[20,30],[50,100]])
    s2 = Segment([[4,6],[15,25],[40,200]])
    s1.difference(s2)
    np.testing.assert_equal(s1._data, np.array([[0,4],[6,10],[15,20],[25,30],[40,50],[100,200]]))

def test_segment_difference_2():
    s1 = Segment([[0,10],[20,30],[50,100]])
    s2 = Segment([[4,6],[15,25],[40,200]])
    s3 = Segment([[8,20],[35,60],[150,180]])
    nose.tools.assert_equal( s1 ^ s2 ^ s3, s1.difference(s2,s3) )


if __name__ == "__main__":
    nose.main()
