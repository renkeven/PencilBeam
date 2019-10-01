"""
Simple rotation method
"""
import numpy as np

class Rotation(object):

    def rot_mat(i,a):
        """
        A crude way of not using dictionaries to return a 3x3 rotation matrix
        0 = x, 1 = y, 2 = z
        """
        if i == 0:
            return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        elif i == 1:
            return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]).astype(float)
        elif i == 2:
            return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]).astype(float)

    rot_mat = staticmethod(rot_mat)

if __name__ == '__main__':
    print(Rotation.rot_mat(1,0.5))


