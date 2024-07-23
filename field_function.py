import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np

class FieldFunction:
    ''' Interpolates a field array to provide a field function (with periodic BCs). '''
    def __init__(self, image_array, interp_method='linear'):
        # store image_array for sanity check
        try: self.image_array=image_array.detach().cpu().numpy()
        except: self.image_array=image_array

        # normalize==True implies: linspace for coords so that it is invariant of the original resolution
        coords = self._make_uniform_coords(image_array.shape, normalized=True, mesh=False)
        self.interpolator=scipy.interpolate.RegularGridInterpolator(coords, self.image_array, method=interp_method)
        #NOTE: 7/22/24, Verified that: self.image_array == self.image_array[*np.meshgrid(*coords)] (this is in fact the definition of mesh_grid)

    def __call__(self, points):
        '''
        Either points.shape==[n_points, n_dim] or points == np.meshgrid(*coords).
        Also this function has period=1 in all dimensions.
        '''
        points = np.asarray(points) # make it periodic
        points -= points.astype(int) # make it periodic
        try: return self.interpolator(points) # points.shape==[n_points, n_dim]
        except ValueError: return self.interpolator(tuple(points)) # or points == np.meshgrid(*coords)

    @property
    def shape(self):
        return self.image_array.shape

    @staticmethod # Verified to work: 7/22/24
    def _make_uniform_coords(shape, normalized=True, mesh=True, flat_mesh=False):
        '''
        Makes uniform coords across the space defined by shape.
        :param shape: shape defining the space
        :param normalized: make coords will span [0,1]
        :param mesh: return a mesh grid (else returns just indep linspace coords)
        :param flat_mesh: return a sequence of all possible coordinates (i.e. flat_mesh.shape==[n_dim, n_points])
        '''
        coord_span = (lambda dim: np.linspace(0,1,dim)) if normalized else np.arange
        indices_ = [coord_span(dim) for dim in shape]
        if not mesh:
            assert not flat_mesh, 'Invalid options!'
            return indices_
        mesh_ = np.meshgrid(*indices_, indexing='ij', copy=False)
        return [coords.ravel() for coords in mesh_] if flat_mesh else mesh_

    def reconstruct_field(self, shape=None, resolution_scale=1, freq_scale=1):
        '''
        Reconstruct the Nd field array
        :param shape: reconstructed field shape
        :param resolution_scale: scales the shape parameter
        :param freq_scale: scales coordinates to make periodic copies of regular field
        '''
        if shape is None: shape=self.image_array.shape
        shape=[dim*resolution_scale for dim in shape]
        mesh = np.stack(self._make_uniform_coords(shape)) #, flat_mesh=True)).T
        mesh *= freq_scale # this will cause periodic sampling by reaching outside the actual bounds of interpolation
        reconstructed = self(mesh).reshape(*shape)
        return reconstructed

    def sanity_check2d(self, **kwd_args):
        reconstructed_tensor = self.reconstruct_field(self.image_array.shape, **kwd_args)
        # Display the original and reconstructed tensors as images
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(self.image_array, cmap='gray')
        ax[0].set_title("Original Field")
        ax[1].imshow(reconstructed_tensor, cmap='gray')
        ax[1].set_title(f"Reconstructed Field from ({self.interpolator.method}) Interp Function")
        plt.show()

        # Assert that it worked (it did! wahoo!)
        if self.image_array.shape==reconstructed_tensor.shape:
            assert (np.abs(self.image_array-reconstructed_tensor)).mean() < 1e-4
            print('Passed equivalence assertion!')

import random

if __name__=='__main__':
    v_train = np.load('v_train.npy')
    field_tensor = random.choice(v_train)
    field_function = FieldFunction(field_tensor, interp_method='cubic')
    print(f'field_function.shape=={field_function.shape}')
    field_function.sanity_check2d(resolution_scale=1)
    field_function.sanity_check2d(resolution_scale=2)
    field_function.sanity_check2d(resolution_scale=2, freq_scale=2)