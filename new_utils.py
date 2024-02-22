"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
import numpy as np
from numpy.typing import NDArray

#1-B
def scale_data(X_bi = NDArray[np.floating]):
   if not issubclass(X_bi.dtype.type, np.floating) or (X_bi < 0).any() or (X_bi > 1).any():
      return False
      
   return True

#1-B
def scale_data_1(y_bi = NDArray[np.int32]):
   if not issubclass(y_bi.dtype.type, np.int32):
      return False

   return True
      
