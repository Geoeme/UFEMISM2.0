import numpy as np
from scipy.spatial import Delaunay



def area_weighted_average(vertices, nodal_values):
    """
    Compute area-weighted average for scattered data using Delaunay triangulation
    
    Parameters:
    vertices : array (N,2) - x,y coordinates of points
    nodal_values : array (N,) - values at each vertex
    
    Returns:
    weighted_avg : float - area-weighted mean
    """
    # Convert to numpy arrays if they're xarray DataArrays
    vertices = np.asarray(vertices)
    nodal_values = np.asarray(nodal_values)
    
    # Remove NaN values (both coordinates and corresponding values)
    valid_mask = ~np.isnan(nodal_values)
    vertices = vertices[valid_mask]
    nodal_values = nodal_values[valid_mask]
    
    # Create Delaunay triangulation
    try:
        tri = Delaunay(vertices)
    except:
        raise ValueError("Delaunay triangulation failed - check for duplicate points or colinearities")
    
    # Get vertex indices for all triangles
    triangles = tri.points[tri.simplices]  # Shape: (M,3,2)
    
    # Vectorized area calculation
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.abs(a[:,0]*b[:,1] - a[:,1]*b[:,0])
    
    # Mean value per triangle
    tri_values = nodal_values[tri.simplices]
    tri_means = np.mean(tri_values, axis=1)
    
    # Weighted average
    total_area = np.sum(areas)
    if total_area == 0:
        return np.nan
    
    return np.sum(areas * tri_means) / total_area


def calculate_area_percentages(V, Hi, ranges):
    """
    Calculate percentage of area for each thickness range.
    
    Args:
        V: Array of vertices (N, 2)
        Hi: Array of thickness values (N,)
        ranges: List of tuples defining ranges [(min1, max1), (min2, max2), ...]
    
    Returns:
        List of area percentages for all ranges
    """
    # Convert inputs and remove NaN values
    vertices = np.asarray(V)
    thickness = np.asarray(Hi)
    valid_mask = ~np.isnan(thickness)
    vertices = vertices[valid_mask]
    thickness = thickness[valid_mask]
    
    # Return zeros if we don't have enough points for triangulation
    if len(vertices) < 3:
        return [0.0] * len(ranges)
    
    try:
        tri = Delaunay(vertices)
    except:
        # Return zeros if triangulation fails
        return [0.0] * len(ranges)
    
    # Get triangles and their properties
    triangles = tri.points[tri.simplices]
    tri_values = thickness[tri.simplices]
    tri_means = np.mean(tri_values, axis=1)
    
    # Calculate triangle areas
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.abs(a[:,0]*b[:,1] - a[:,1]*b[:,0])
    total_area = np.sum(areas)
    
    if total_area == 0:
        return [0.0] * len(ranges)
    
    # Calculate area percentages for each range
    percentages = []
    for min_val, max_val in ranges:
        in_range = (tri_means >= min_val) & (tri_means < max_val)
        range_area = np.sum(areas[in_range])
        percentages.append((range_area / total_area) * 100)
    
    return percentages

