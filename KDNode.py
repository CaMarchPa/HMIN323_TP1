class KDNode:
    def __init__(self, pivot = None, axis = None, left_child = None, right_child = None):
        self.pivot       = pivot
        self.axis        = axis
        self.left_child  = left_child
        self.right_child = right_child
    
    def view_kdtree(kdtree, bbox=[[-1., 1.],[-1., 1.],[-1., 1.]], radius=0.05):
    import numpy as np
    import openalea.plantgl.all as pgl

    scene = pgl.Scene()
    sphere = pgl.Sphere(radius,slices=16,stacks=16)
    silver = pgl.Material(ambient=(49,49,49),diffuse=3.,specular=(129,129,129),shininess=0.4)
    gold = pgl.Material(ambient=(63,50,18),diffuse=3.,specular=(160,141,93),shininess=0.4)

    if isinstance(kdtree, KDNode):
        dim = kdtree.axis
        plane_bbox = [b for i,b in enumerate(bbox) if i != dim]
        plane_points = []
        plane_points += [np.insert([plane_bbox[0][0],plane_bbox[1][0]],dim,kdtree.pivot[dim])]
        plane_points += [np.insert([plane_bbox[0][0],plane_bbox[1][1]],dim,kdtree.pivot[dim])]
        plane_points += [np.insert([plane_bbox[0][1],plane_bbox[1][1]],dim,kdtree.pivot[dim])]
        plane_points += [np.insert([plane_bbox[0][1],plane_bbox[1][0]],dim,kdtree.pivot[dim])]

        left_bbox = np.copy(bbox).astype(float)
        right_bbox = np.copy(bbox).astype(float)
        left_bbox[dim,1] = kdtree.pivot[dim]
        right_bbox[dim,0] = kdtree.pivot[dim]

        scene += pgl.Shape(pgl.Translated(kdtree.pivot,sphere),gold)
        scene += view_kdtree(kdtree.left_child, bbox=left_bbox, radius=radius)
        scene += view_kdtree(kdtree.right_child, bbox=right_bbox, radius=radius)
        scene += pgl.Shape(pgl.Polyline(plane_points+[plane_points[0]],width=2),pgl.Material(ambient=(0,0,0)))
        scene += pgl.Shape(pgl.FaceSet(plane_points,[range(4)]),pgl.Material(ambient=(0,0,0),transparency=0.6))

    else:
        assert (type(kdtree) == list) or (isinstance(kdtree,np.ndarray))
        for p in kdtree:
            scene += pgl.Shape(pgl.Translated(p,sphere),silver)

    return scene
    
	def view_kdtree(kdtree, bbox=[[-1., 1.],[-1., 1.],[-1., 1.]], radius=0.05):
    import numpy as np
    import openalea.plantgl.all as pgl

    scene = pgl.Scene()
    sphere = pgl.Sphere(radius,slices=16,stacks=16)
    silver = pgl.Material(ambient=(49,49,49),diffuse=3.,specular=(129,129,129),shininess=0.4)
    gold = pgl.Material(ambient=(63,50,18),diffuse=3.,specular=(160,141,93),shininess=0.4)

    if isinstance(kdtree, KDNode):
        dim = kdtree.axis
        plane_bbox = [b for i,b in enumerate(bbox) if i != dim]
        plane_points = []
        plane_points += [np.insert([plane_bbox[0][0],plane_bbox[1][0]],dim,kdtree.pivot[dim])]
        plane_points += [np.insert([plane_bbox[0][0],plane_bbox[1][1]],dim,kdtree.pivot[dim])]
        plane_points += [np.insert([plane_bbox[0][1],plane_bbox[1][1]],dim,kdtree.pivot[dim])]
        plane_points += [np.insert([plane_bbox[0][1],plane_bbox[1][0]],dim,kdtree.pivot[dim])]

        left_bbox = np.copy(bbox).astype(float)
        right_bbox = np.copy(bbox).astype(float)
        left_bbox[dim,1] = kdtree.pivot[dim]
        right_bbox[dim,0] = kdtree.pivot[dim]

        scene += pgl.Shape(pgl.Translated(kdtree.pivot,sphere),gold)
        scene += view_kdtree(kdtree.left_child, bbox=left_bbox, radius=radius)
        scene += view_kdtree(kdtree.right_child, bbox=right_bbox, radius=radius)
        scene += pgl.Shape(pgl.Polyline(plane_points+[plane_points[0]],width=2),pgl.Material(ambient=(0,0,0)))
        scene += pgl.Shape(pgl.FaceSet(plane_points,[range(4)]),pgl.Material(ambient=(0,0,0),transparency=0.6))

    else:
        assert (type(kdtree) == list) or (isinstance(kdtree,np.ndarray))
        for p in kdtree:
            scene += pgl.Shape(pgl.Translated(p,sphere),silver)

    return scene
