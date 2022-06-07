# UV_Unwrap
unwrap vertex attribute into UV map


Given a Wavefront Obj file with UV coordinate, the UV indices to vertex indices correspondence could be built. With this correspondence and UV coordinates, we can get a UV map from vertex attribute array, e.g., get texture map from vertex color array, get position map from vertex coordinates. 

UV_unwrapper.py shows a demo where we build a position map from vertex coordinates and recover to the vertex coordinates from position map. 

