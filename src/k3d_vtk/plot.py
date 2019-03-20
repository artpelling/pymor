import IPython
from ipywidgets import IntSlider, interact, widgets

from k3d_vtk.reader import read_vtkfile
from k3d.plot import Plot as k3dPlot
import k3d
from vtk.util import numpy_support
import numpy as np
import vtk


def _transform_to_k3d(timestep, poly_data, color_attribute):
    '''
    this function mirrors the prepartion in k3d.vtk_poly_data
    :param timestep: attribute from vtk collection file
    :param poly_data: vtk reader Output for one single vtk file
    :param color_attribute: Determines mesh colorization, 3-Tuple of Vtk Dataset name, min value, max value
    :return: 5-Tuple to match necessary updates to mesh when advancing the timestep
    '''
    if poly_data.GetPolys().GetMaxCellSize() > 3:
        cut_triangles = vtk.vtkTriangleFilter()
        cut_triangles.SetInputData(poly_data)
        cut_triangles.Update()
        poly_data = cut_triangles.GetOutput()

    if color_attribute is not None:
        attribute = numpy_support.vtk_to_numpy(poly_data.GetPointData().GetArray(color_attribute[0]))
        color_range = color_attribute[1:3]
    else:
        attribute = []
        color_range = []
    vertices = numpy_support.vtk_to_numpy(poly_data.GetPoints().GetData())
    indices = numpy_support.vtk_to_numpy(poly_data.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]

    return timestep, np.array(attribute, np.float32), color_range, \
        np.array(vertices, np.float32), np.array(indices, np.uint32)


class VTKPlot(k3dPlot):
    def __init__(self, vtk_data, vmin, vmax, model_matrix=None, color_map=k3d.basic_color_maps.CoolWarm,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.idx = 0

        if 'transform' in kwargs.keys() and len(vtk_data) > 1:
            raise RuntimeError('supplying transforms is currently not supported for teim series VTK Data')

        color_attribute = ('Data', vmin, vmax)
        self.mesh = k3d.vtk_poly_data(vtk_data[0][1], color_attribute=color_attribute,
                                      color_map=color_map)
        self.vtk_data = [_transform_to_k3d(v[0], v[1], color_attribute) for v in vtk_data]
        self.timestep = vtk_data[0][0]
        self += self.mesh

    def _goto_idx(self, idx):
        if not (0<= self.idx < len(self.vtk_data)):
            raise RuntimeWarning(f'Index {idx} outside data range for VTKPlot')
            return
        self.idx = idx
        self.timestep, self.mesh.attribute, self.mesh.color_range, \
            self.mesh.vertices, self.mesh.indices = self.vtk_data[self.idx]

    def dec(self):
        self._goto_idx(self.idx-1)

    def inc(self):
        self._goto_idx(self.idx+1)


def plot(vtkfile_path, vmin, vmax):
    data = read_vtkfile(vtkfile_path)
    size = len(data)

    xmin, xmax, ymin, ymax = 0, 1, 0, 1
    zmin, zmax = 0, 0

    # guesstimate
    fov_angle = 30
    absx = np.abs(xmax - xmin)
    # camera[posx, posy, posz, targetx, targety, targetz, upx, upy, upz]
    c_dist = np.sin((90-fov_angle)*np.pi/180) * absx / (2 * np.sin(fov_angle*np.pi/180))

    plot = VTKPlot(data, vmin, vmax, grid_auto_fit=False, camera_auto_fit=False)
    # display needs to have been called before chaning camera
    plot.display()
    plot.grid = (xmin, ymin, zmin, xmax, ymax, zmax)
    plot.camera = ((xmax+xmin)/2,(ymax+ymin)/2,(zmax+zmin)/2 + c_dist,
                   (xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2,
                   0, 1, 0)
    if size > 1:
        ws = interact(idx=IntSlider(min=0, max=size-1, step=1, value=0, description='Timestep:')).widget(plot._goto_idx)
        IPython.display.display(ws)
    return plot
