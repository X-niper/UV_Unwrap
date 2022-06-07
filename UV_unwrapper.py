import torch 
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes, packed_to_list
from pytorch3d.renderer.mesh.textures import TexturesUV, TexturesVertex
from scipy.interpolate import griddata, interp2d
import numpy as np 
import cv2 


def convert_to_textureVertex(textures_uv: TexturesUV, meshes:Meshes) -> TexturesVertex:
    verts_colors_packed = torch.zeros_like(meshes.verts_packed())
    verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()  # (*)
    return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()))

class UV_unwrapper:
    def __init__(self, template_obj_path:str, target_size=4096) -> None:
        verts, face_idx, aux = load_obj(template_obj_path)
        mesh = load_objs_as_meshes([template_obj_path])
        face_uv_idx = face_idx.textures_idx
        face_vert_idx = face_idx.verts_idx
        uv_idx_to_vert_idx_tbl = {}
        vert_idx_to_uv_idx_tbl = {}

        for i in range(len(face_uv_idx)):
            uv_i = face_uv_idx[i]
            vert_i = face_vert_idx[i]
            for j in range(3):
                vert_idx = int(vert_i[j])
                uv_idx = int(uv_i[j])
                if uv_idx not in uv_idx_to_vert_idx_tbl:
                    uv_idx_to_vert_idx_tbl[uv_idx] = vert_idx
                if vert_idx not in vert_idx_to_uv_idx_tbl:
                    vert_idx_to_uv_idx_tbl[vert_idx] = uv_idx
        print(f'uv_idx_to_vert_idx_tbl built, total {len(uv_idx_to_vert_idx_tbl)}')

        vert_idx_sorted_according_to_uv_idx = []
        for uv_idx in sorted(uv_idx_to_vert_idx_tbl.keys()):
            vert_idx_sorted_according_to_uv_idx.append(uv_idx_to_vert_idx_tbl[uv_idx])
        uv_idx_sorted_according_to_vert_idx = []
        for vert_idx in sorted(vert_idx_to_uv_idx_tbl.keys()):
            uv_idx_sorted_according_to_vert_idx.append(vert_idx_to_uv_idx_tbl[vert_idx])

        
        self.vert_idx_sorted_according_to_uv_idx = vert_idx_sorted_according_to_uv_idx 
        self.uv_idx_sorted_according_to_vert_idx = uv_idx_sorted_according_to_vert_idx 
        self.uv_idx_to_vert_idx_tbl = uv_idx_to_vert_idx_tbl 
        self.uv_coordinates = aux.verts_uvs.cpu().numpy()
        self.vertex_num = len(verts)
        self.target_size = target_size
        xi = yi = np.arange(0, target_size, 1) + 0.5 
        xi, yi = np.meshgrid(xi, yi)
        yi = yi[::-1, :]
        self.grid_x = xi.astype(np.float32)
        self.grid_y = yi.astype(np.float32)
        self.scaled_uv = (self.uv_coordinates * target_size).astype(np.float32)

        self.faces_uvs_tensor = face_idx.textures_idx[None]
        self.verts_uvs_tensor = aux.verts_uvs[None]
        self.mesh = mesh 
        self.get_auxiliary_points(dist=10)
    

    def get_auxiliary_points(self, dist:float=10):
        a0, a1, a2 = -dist, self.target_size // 2, self.target_size + dist 
        self.auxiliary_uv = np.array([
            [a0, a0],
            [a1, a0],
            [a2, a0],
            [a2, a1],
            [a2, a2],
            [a1, a2],
            [a0, a2],
            [a0, a1]
        ]).astype(np.float32)
        self.auxiliary_attribute = np.repeat(np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1,3), 8, axis=0)
        return 

    def get_attribute_map(self, verts_attribute:np.ndarray, method='linear', fill_value=None, dtype=np.float32):
        ''' unwrap vertex attribute to uv map 
            attribute should be of shape (V, None)
        '''
        if not isinstance(verts_attribute, np.ndarray):
            raise TypeError('attribute should be numpy array')
        if verts_attribute.shape[0] != self.vertex_num:
            raise ValueError('the first dim should be equal to vertex num')
        
        sorted_attribute = verts_attribute[self.vert_idx_sorted_according_to_uv_idx]
        sorted_attribute = np.concatenate([sorted_attribute, self.auxiliary_attribute], axis=0)
        scaled_uv = np.concatenate([self.scaled_uv, self.auxiliary_uv], axis=0)
        attribute_map = []
        for i in range(sorted_attribute.shape[1]):
            if fill_value is None:
                fill_value = sorted_attribute[:, i].min()
            attribute_map_i = griddata(scaled_uv, sorted_attribute[:, i], (self.grid_x, self.grid_y), method=method, fill_value=fill_value)
            attribute_map.append(attribute_map_i)
        attribute_map = np.stack(attribute_map, axis=2)
        return attribute_map.astype(dtype)

    def get_attribute_array_by_pytorch3d(self, attribute_map:np.ndarray):
        attribute_map_tensor = torch.Tensor(attribute_map)
        texture_uv = TexturesUV(maps=[attribute_map_tensor], faces_uvs=self.faces_uvs_tensor, verts_uvs=self.verts_uvs_tensor)
        verts_colors_packed = torch.zeros_like(self.mesh.verts_packed())
        verts_colors_packed[self.mesh.faces_packed()] = texture_uv.faces_verts_textures_packed()  # (*)
        # print(verts_colors_packed.max(), verts_colors_packed.min())
        return verts_colors_packed.cpu().numpy()


    def get_attribute_array_by_scipy(self, attribute_map:np.ndarray, kind='linear'):
        # todo convert UV index to vertex index 
        x = self.scaled_uv[:,0]
        y = self.scaled_uv[:,1]

        attribute_array = []
        for channel in range(attribute_map.shape[-1]):
            f = interp2d(self.grid_x[0], self.grid_y[:,0], attribute_map[...,channel], kind=kind)
            interpolated_value = []
            for xi, yi in zip(x, y):
                interpolated_value.append(f(xi,yi))
            interpolated_value = np.array(interpolated_value)
            attribute_array.append(interpolated_value)
        
        attribute_array = np.concatenate(attribute_array, axis=1)
        print(attribute_array.max(), attribute_array.min())
        return attribute_array[self.uv_idx_sorted_according_to_vert_idx]



class Mesh_attribute_processor:
    def __init__(self, template_obj_path:str='', rescale_factor:float=None, offset_factor:float=None) -> None:
        if rescale_factor is not None:
            self.rescale_factor = rescale_factor
            self.offset_factor = offset_factor
        else:
            verts, face_idx, aux = load_obj(template_obj_path)
            verts = verts.cpu().numpy()
            print(np.max(verts, axis=0), np.min(verts, axis=0))
            self.rescale_factor = 1 / (np.max(verts) - np.min(verts))
            self.offset_factor = 0.5 

    def get_rescaled_verts(self, obj_path:str)->np.ndarray:
        verts, _, _ = load_obj(obj_path, load_textures=False)
        verts = verts.cpu().numpy()
        verts = verts * self.rescale_factor + self.offset_factor
        if np.any(verts<0):
            print('warning, negative value detected')
        return verts.astype(np.float32)



    

        
        
if __name__ == "__main__":
    template_obj_path = 'test.obj'
    test_obj_path = template_obj_path
    unwrapper = UV_unwrapper(template_obj_path, target_size=4096)
    mesh_attribute_processor = Mesh_attribute_processor(rescale_factor=1/300, offset_factor=0.5)

    verts_rescaled = mesh_attribute_processor.get_rescaled_verts(test_obj_path)
    print(verts_rescaled.max(), verts_rescaled.min())

    position_map = unwrapper.get_attribute_map(verts_rescaled, method='cubic', fill_value=0)
    print(position_map.max(), position_map.min())
    cv2.imwrite('position_map.png', position_map* 255.5)
    position_map_512 = cv2.resize(position_map* 255.5, (512,512))
    cv2.imwrite('position_map_512.png', position_map_512)

    interpolated_array_scipy = unwrapper.get_attribute_array_by_scipy(position_map, kind='linear')
    interpolated_array_torch = unwrapper.get_attribute_array_by_pytorch3d(position_map)

    diff_torch = interpolated_array_torch - verts_rescaled
    diff_scipy = interpolated_array_scipy - verts_rescaled

    diff_torch = diff_torch / verts_rescaled
    diff_scipy = diff_scipy / verts_rescaled
    print(np.abs(diff_torch).max(), np.abs(diff_torch).min())
    print(np.abs(diff_scipy).max(), np.abs(diff_scipy).min())

    # import matplotlib.pyplot as plt 
    # plt.hist(diff_torch, bins=30, )
    # plt.savefig('torch.png')
    # plt.hist(diff_scipy, bins=30, )
    # plt.savefig('scipy.png')

    print('done')



        