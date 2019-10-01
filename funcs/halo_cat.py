"""
All the catalog manipulation required to rotate box, p. boundary conditions done here.
"""

import numpy as np
import itertools

class haloCatalog(object):
    """
    Input is some array that is of (n,4) size. (Mass, x, y, z).
    """

    def __init__(self, halobox, h, box_size, little_h = False):
        """
        ensure halobox is some (n,4) array (or some other sized correctly organised array) in the format of (mass, x, y, z).
        """
        self.halobox = halobox
        self.mass = self.halobox[:,0]
        self.position = self.halobox[:,1:4]

        self._x = None
        self._y = None
        self._z = None
        self._vanilla_position = None
    
        self.count = len(self.halobox)

        self.h = h
        self.box_size = box_size

        if little_h == False:
            self.box_size = self.box_size/self.h

        return

    @property
    def vanilla_position(self):
        if self._vanilla_position is None:
            self._vanilla_position = self.halobox[:,1:4]

        return self._vanilla_position

    @property
    def x(self):
        if self._x == None:
           self._x = self.halobox[:,1]

        return self._x

    @property
    def y(self):
        if self._y == None:
            self._y = self.halobox[:,2]

        return self._y

    @property
    def z(self):
        if self._z == None:
            self._z = self.halobox[:,3]

        return self._z 

    def halo_shifter(self,coordinate):
        """
        Takes your box and shifts it in units of box_size in a specified direction. Used as part of creating an 'Rubik's cube' of our simulation to represent p.b.c.
        Coordinates are specificed as (x,y,z) -> [0,1,1] implies shifting y and z by box_size in the positive direction.
        Returns our shifted array.
        """
        assert len(coordinate) == 3, 'Coordinate array needs to be length of 3!'
        
        shifted_position = np.zeros(np.shape(self.position))
        
        for i in range(len(coordinate)):
            shifted_position[:,i] = self.position[:,i] + (self.box_size * coordinate[i])
        
        return shifted_position

    def rubik_cube(self,clone_number):
        """
        Creates an expanded catalogue for periodic boundary conditions lightbeams.
        clone_number is how many extra boxes tagged onto a dimension. clone_number = 1 -> 3x3x3 with your original in the center
        Returns the full catalogue with new positions
        """ 

        #assert (clone_number > 0) & (type(clone_number) == int), 'clone_number needs to be an int > 0!'

        if clone_number == 0:
            return

        permutation = np.linspace(-clone_number, clone_number, 2*clone_number + 1)   
        rubik_cubed = np.copy(self.position)

        for p in itertools.product(permutation,repeat=3):
            if not (self.position == self.halo_shifter(p)).all():
                rubik_cubed = np.concatenate((rubik_cubed,self.halo_shifter(p)),axis=0)

        self.position = rubik_cubed

        return

    def apply_log_mass(self):
        """
        Convert self.mass into log10
        """
        assert np.min(self.mass) > 1e2, 'Appears that you are already in units of logM'

        logmass = np.log10(self.mass)

        self.mass = logmass

        return

    def apply_standard_mass(self):
        """
        Convert self.mass into M (unlog it)
        """
        assert np.max(self.mass) < 1e2, 'Appears that you are already in the usual non-loged units'

        self.mass = 10**(self.mass)

        return

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

    def random_beam_directions():
        """
        First is beam direction along dimension, f_dir is first rotation axis, s_dir is second rotation axis
        """
        beam_direction,f_dir,s_dir = np.random.choice([0,1,2],3,replace=False)

        return beam_direction, f_dir, s_dir

    random_beam_directions = staticmethod(random_beam_directions)

    def cube_rotation(self, f_dir, s_dir, theta, phi, point_rotation=np.array([None,None,None])):
        """
        Rotate the catalog. Point rotation is a subset of cube-rotation where a point (coordinate) is specified instead of the halo catalog
        """
        assert len(point_rotation) == 3, 'If performing a point rotation, point needs to be a vector of length 3'

        if (point_rotation == None).all():
            centered_cat = self.position.T - self.box_size/2.
            first_rotation = np.dot(self.rot_mat(f_dir,theta), centered_cat)
            second_rotation = np.dot(self.rot_mat(s_dir,phi), first_rotation)
            uncentered_cat = second_rotation.T + self.box_size/2.

        else:
            centered_cat = point_rotation.T - self.box_size/2.
            first_rotation = np.dot(self.rot_mat(f_dir,theta), centered_cat)        
            second_rotation = np.dot(self.rot_mat(s_dir,phi), first_rotation)
            uncentered_cat = second_rotation.T + self.box_size/2.

            assert len(uncentered_cat) == 3, 'Check if a vector is the output'

        return uncentered_cat


    def light_beams(self, width, depth, length, theta, phi, verbose=True, *index):
        """
        """
        direction_dict = {0:'x',1:'y',2:'z'}
        beam_direction, f_dir, s_dir = self.random_beam_directions()
        rotated_cat = self.cube_rotation(f_dir, s_dir, theta, phi)

        if index:
            if verbose:
                print('Pencil beam: Index number of halo supplied')
            x0 = rotated_cat[index[0],0]
            y0 = rotated_cat[index[0],1]
            z0 = rotated_cat[index[0],2]

        else:
            if verbose:
                print('Pencil beam: Randomised coordinates')
            random_coordinate = np.random.random_sample(3) * self.box_size
            x0, y0, z0 = self.cube_rotation(f_dir, s_dir, theta, phi, point_rotation = random_coordinate)

        if verbose:
            print(' ')
            print(':::::')
            print(f'Beam Direction is in the {direction_dict[beam_direction]} direction')
            print(f'Rotation around {direction_dict[f_dir]} axis, then {direction_dict[s_dir]} axis')
            print(f'1st, 2nd rotation angles: {theta:.3f}, {phi:.3f}')
            print(f'Initial coordinates ({x0:.2f}, {y0:.2f}, {z0:.2f}) (After Rotation)')
            print(':::::')
            print(' ')        

        initial_coordinate = np.array([x0,y0,z0])

        recovered_index = np.where((rotated_cat[:,beam_direction] < initial_coordinate[beam_direction] + length/2.) & \
                                   (rotated_cat[:,beam_direction] > initial_coordinate[beam_direction] - length/2.) & \
                                   (rotated_cat[:,f_dir] < initial_coordinate[f_dir] + width/2.) & \
                                   (rotated_cat[:,f_dir] > initial_coordinate[f_dir] - width/2.) & \
                                   (rotated_cat[:,s_dir] < initial_coordinate[s_dir] + depth/2.) & \
                                   (rotated_cat[:,s_dir] > initial_coordinate[s_dir] - depth/2.))
                                
        cx0, cy0, cz0 = self.cube_rotation(s_dir, f_dir, -phi, -theta, point_rotation = initial_coordinate)

        #Modulo to obtain objects with the matching indices within the original cube

        modulo_index = np.mod(recovered_index,self.count)

        return modulo_index[0], (cx0, cy0, cz0), (beam_direction, f_dir, s_dir)

    def check_catalog_integrity(self, LO=False):
        """
        Sanity check/tests for your catalog object.
        Checks the units of catalog is in terms of log10 M 
        Checks for position of halos relative to box size. Produces a warning for suspicious activity
        """

        assert np.max(self.vanilla_position) <= self.box_size, 'Catalog\'s halo position exceeds specified box size. Check for h\'s'

        if np.max(self.position)*1.1 < self.box_size:
            print('""""""""""')
            print('Warning: Specified box size is >10% greater than the max position coordinate of catalog. Possible mismatch in specified box parameters/catalog size')
            print('""""""""""')
            print('')

        if LO==True:
            assert (np.max(self.mass) < 1e2) | (np.max(self.mass) < 5), 'Mass HIGHLY not likely to be in log units.'

        else:
            if np.max(self.mass) < 1e2:
                print('Mass is likely to be in log units')

        if np.max(self.mass) >= 1e2:
            print('Mass is likely to be in standard (non-logged) units')

        clone_number = int(((len(self.position)/len(self.vanilla_position))**(1/3) - 1)/2)

        print(f'Catalog is cloned {clone_number:0} times in a direction')
        print('')
        print('')
        return

if __name__ == '__main__':
    """
    Test: Create Rubiks cube, Rotate, Pencilbeam, Spit random index
    """
    np.set_printoptions(threshold=np.inf)
    halo_index = 7000

    fakecatalog = np.genfromtxt('fake_catalog')
    h, box_size = np.genfromtxt('params')

    test = haloCatalog(fakecatalog, h, box_size)
    test.rubik_cube(1)

    test.check_catalog_integrity()

    beam_width = 30
    beam_depth = 30
    beam_length = 1200
    theta = np.random.choice([-1.,1.])*((np.pi/4.*np.random.random()) + np.pi/6.)
    phi = np.random.choice([-1.,1.])*((np.pi/4.*np.random.random()) + np.pi/6.)

    print('boxsize', test.box_size)
    print('index of max mass', np.argmax(test.mass), '+ total number of objects',test.count)

    index, center, _ = test.light_beams(30, 30, 1200, theta, phi, True, halo_index)

    print(index)
    print(center, test.vanilla_position[halo_index])





