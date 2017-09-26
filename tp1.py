from KDNode import KDNode;
import openalea.plantgl.all as pgl

def createkdtree(point_list, minbucketsize = 3, int depth = 0):
	
	if len(point_list) > 2*minbucketsize :
		axis = depth%3
		sorted(point_list, key= lambda x : x[axis])
		p_med = point_list[len(point_list)//2]
		right_list = pointlist[:len(point_list)]
		left_list = pointlist[len(point_list)+1:]
		node = KDNode(p_med, axis, left_list, right_list)
		createkdtree(left_list, minbucketsize, depth+1)
		createkdtree(right_list, minbucketsize, depth+1)
		return node
	else
		return point_list

def closestpoint(kdtree, point ):
	from openalea.plantgl.all import norm
	if isinstance(kdtree, KDNode):
		axis = dktree.axis
		pcoord = point[axis]
		if pcoord = kdtree.pivot[axis]:
			insidesubdiv, oppositesubdiv = kdtree.left_child, kdtree.right_child
		else:
			insidesubdiv, oppositesubdiv = kdtree.right_child, kdtree.left_child
			
		####
		candidat = closestpoint(insidesubdiv, point)
		
		if norm(point-candidat) > abs(pcoord-kdtree.pivot[axis]):
			if norm(point-kdtree.pivot) < norm(point-candidat):
			candidat = kdtree.pivot
		alternative_candidat = closestpoin(kdtree, oppositesubdiv)
	

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
    
	def print_kdtree(kdtree, depth = 0):
		if isinstance(kdtree, KDNode):
			print ('  '*depth) + 'Node :', kdtree.axis,  kdtree.pivot
			print_kdtree(kdtree.left_child, depth+1)
        	print_kdtree(kdtree.right_child, depth+1)
		else:
			assert (type(kdtree) == list) or (isinstance(kdtree,np.ndarray))
			print ('  '*depth) + 'Leaf :', kdtree
			
	def brute_force_closest(point, pointlist):
		""" Find the closest points of 'point' in 'pointlist' using a brute force approach """
		import sys
		pid, d = -1, sys.maxint
		for i, p in enumerate(pointlist):
			nd = pgl.norm(point-p) 
			if nd &lt; d:
				d = nd
				pid = i
		return pointlist[pid]

	def generate_random_point(size=[1,1,1], distribution='uniform'):
		from random import uniform, gauss
		if distribution == 'uniform':
			return pgl.Vector3(uniform(-size[0],size[0]), uniform(-size[1],size[1]), uniform(-size[2],size[2])) 
		elif distribution == 'gaussian':
			return pgl.Vector3(gauss(0,size[0]/3.), gauss(0,size[1]/3.), gauss(0,size[1]/3.)) 

	def generate_random_pointlist(size=[1,1,1], nb = 100, distribution='uniform'):
		return [generate_random_point(size, distribution=distribution) for i in xrange(nb)]

	def test_kdtree(create_kdtree_func, closestpoint_func, nbtest=100, nbpoints=1000, size=[1,1,1], minbucketsize=2):
		import time

		points = generate_random_pointlist(nb = nbpoints, size=size, distribution='uniform')
		mkdtree = create_kdtree_func(points, minbucketsize)
		pgl.Viewer.display(view_kdtree(mkdtree, radius=0.03, bbox=[[-float(s),float(s)] for s in size]))
		kdtime, bftime = 0,0
		for i in xrange(nbtest):
			testpoint = generate_random_point(size)
			t = time.time()
			kpoint = closestpoint_func(testpoint, mkdtree)
			kdtime += time.time()-t
			t = time.time()
			bfpoint = brute_force_closest(testpoint, points)
			bftime += time.time()-t
			if kpoint != bfpoint: 
				raise ValueError('Invalid closest point')
		print 'Comparative execution time : KD-Tree [', kdtime,'], BruteForce [', bftime,']'

		return kdtime, bftime
	
	def plot_execution_time(nbpoints_min=10, nbpoints_max=5000):
		import matplotlib.pyplot as plt
		
		kd_times = []
		bf_times = []
		nb_points = range(nbpoints_max, nbpoints_min, 10)
		
		for n in nb_points:
			kdtime, bf_time = test_kdtree(createkdtree, closestpoint, nbpoints=n)
			kd_times += [kdtime]
			bf_times += [bftime]
			
		plt.figure("Execution Time")
		plt.plot(nb_points, kd_times, color='r', label='KD-Tree')
		plt.plot(nb_points, bf_times, color='b', label='Brute Force')
		plt.legend()
		plt.show()
		
		
