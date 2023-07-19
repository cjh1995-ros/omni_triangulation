"""
title: 'Direct Triangulation with Spherical Projection for Omnidirectional Cameras'
author: Ciar ́an Eising
url : https://arxiv.org/pdf/2206.03928.pdf
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def ematrix(rmat: np.ndarray,
            tvec: np.ndarray,
            )-> np.ndarray:
    t1, t2, t3 = tvec[0], tvec[1], tvec[2]
    tmat = np.array([
        [0, t3, -t2],
        [-t3, 0, t1],
        [t2, -t1, 0]
    ], dtype=np.float32)

    return rmat @ tmat

def is_in_egeometry(u1: np.ndarray,
                    u2: np.ndarray,
                    emat: np.ndarray,
                    )->np.ndarray:
    # u1, u2 = (3, )
    val = u1 @ emat @ u2
    if np.abs(val) < 1e-8: return True  # close to 0
    return False

def midpoint(u1: np.ndarray,
             u2: np.ndarray,
             C1: np.ndarray = np.zeros((3, ), dtype=np.float32),
             C2: np.ndarray = np.zeros((3, ), dtype=np.float32),
    ) -> np.ndarray:
    """
        Mid point is for triangulate a point with two unit vectors.
        
        Original fomula:
            - X = (C1 + C2 + alpha1 * u1 + alpha2 * u2) / 2
            
        Parameters description:
            - alpha1, alpha2: coefficient of unit vector.
            - u1, u2: unit vector in 3d euclidean field
            - C1, C2: position of unit sphere, they don't have to be fit in origin for now.
    """
    # calculate value of alphas
    t = C2 - C1
    
    alpha1_numer = (u2.T @ u2) * (u1.T @ t) - (u1.T @ u2) * (u2.T @ t)
    alpha1_denom = (u1.T @ u1) * (u2.T @ u2) - (u1.T @ u2) * (u1.T @ u2)
    alpha1 = alpha1_numer / alpha1_denom

    alpha2_numer = (u1.T @ u2) * (u1.T @ t) - (u1.T @ u1) * (u2.T @ t)
    alpha2_denom = (u1.T @ u1) * (u2.T @ u2) - (u1.T @ u2) * (u1.T @ u2)
    alpha2 = alpha2_numer / alpha2_denom

    X = (C1 + C2 + alpha1 * u1 + alpha2 * u2) / 2
    
    return X

def sph_lin(us: np.ndarray,
            Rs: np.ndarray,
            Cs: np.ndarray,
    ) -> np.ndarray:
    """
        Spherical linear triangulation method. Just using linear method for 
        estimating position of point. 

        Parameters description:
            - us: The set of unit sphere vector. Shape is (n, 3)
            - Rs: The set of orientation matrix. Shape is (n, 3, 3)
            - Cs: The set of camera position. Shape is (n, 3)
    """
    
    A = []
    b = []

    for (u, R, C) in zip(us, Rs, Cs):
        u1, u2, u3 = u[0], u[1], u[2]
        r1, r2, r3 = R.T[0, :], R.T[1, :], R.T[2, :]
        
        _A = np.array([
            u1 * r3 - u3 * r1,
            u2 * r3 - u3 * r2
        ], dtype=np.float64)
        
        _b = np.array([
            [(u1 * r3 - u3 * r1).T @ C],
            [(u2 * r3 - u3 * r2).T @ C],
        ], dtype=np.float64)

        A.append(_A)
        b.append(_b)
    
    A = np.vstack(A)
    b = np.vstack(b)
    X = np.linalg.pinv(A) @ b

    return X.T


def sph_quad(u1: np.ndarray,
             u2: np.ndarray,
             R1: np.ndarray,
             R2: np.ndarray,
             C1: np.ndarray,
             C2: np.ndarray,
    ) -> np.ndarray:
    """
        Direct triangulation methods. In this function, it minimized
        the sum of squares.

        Parameters description:
            - u1, u2: Unit vector in 3d euclidean field
            - Rt1, Rt2: Orientation + Position
            - e: Transform from C1 to C2
            - Rb: Rotation vector which convert e to (1, 0, 0).
            - n: Normal vector of epipolar geometry, if u12 > u13 (0, lambda, 1) else (0, 1, lambda)  
            - lambda: Component of n vector. made from a, b, c 
            - a, b, c: Coefficient of distance function.
    """
    # Get rotation matrix which convert C1 to C2
    Rr = R1.T @ R2
    
    # Get Rb matrix
    e = C2 - C1
    x = np.array([1, 0, 0], dtype=np.float32)
    Rb_vec = np.cross(x, e)
    Rb = R.from_rotvec(Rb_vec).as_matrix()

    # check whether two vector in egeomtry now
    emat = ematrix(Rr, e)
    if is_in_egeometry(u1, u2, emat): raise Exception("Already in same Epipolar geometry")

    # Convert u1, u2 from world coord to new coord
    u1 = Rb @ u1
    u2 = Rb @ Rr @ u2

    # extract component of unit vectors
    u12, u13 = u1[1], u1[2]
    u22, u23 = u2[1], u2[2]

    # calculate a, b, c
    a = u13 * u13 + u23 * u23 if np.abs(u12) > np.abs(u13) else u12 * u12 + u22 * u22
    b = 2 * (u12 * u13 + u22 * u23)
    c = u12 * u12 + u22 * u22 if np.abs(u12) > np.abs(u13) else u13 * u13 + u23 * u23

    if np.abs(b) <= 1e-8: raise Exception("No validate triangulate points")
    
    # calculate lambda
    lamb_numer = c - a - np.sqrt((a - c) * (a - c) + b * b)
    lamb = lamb_numer / b

    # calculate n vector 
    n = np.array([0, lamb, 1], dtype=np.float32) if np.abs(u12) > np.abs(u13) else np.array([0, 1, lamb], dtype=np.float32)

    n_norm = np.linalg.norm(n)
    nn = n_norm * n_norm

    # project unit vectors in sphere to epipolar geometry
    u1 = u1 - (u1.T @ n) / nn * n
    u2 = u2 - (u2.T @ n) / nn * n

    e_mat = np.array([
        [0, e[2], -e[1]],
        [-e[2], 0, e[0]],
        [e[1], -e[0], 0]
    ], dtype=np.float32)
    E = Rr @ e_mat

    # if np.abs(u1 @ E @ u2) > 1e-7: raise Exception("Not in Epipolar constraint")
    

    # Calculate point in C1 coord
    point = midpoint(u1, u2, C2 = e)

    # transform point in origin coord
    # First Rb -> R1
    point = Rb.T @ point + C1
    
    return point


def sph_abs(u1: np.ndarray,
             u2: np.ndarray,
             R1: np.ndarray,
             R2: np.ndarray,
             C1: np.ndarray,
             C2: np.ndarray,
    ) -> np.ndarray:
    """
        Direct triangulation methods. In this function, it minimized
        the sum of magnitude.

        Parameters description:
            - u1, u2: Unit vector in 3d euclidean field
            - Rt1, Rt2: Orientation + Position
            - e: Transform from C1 to C2
            - Rb: Rotation vector which convert e to (1, 0, 0).
            - n: Normal vector of epipolar geometry, if u12 > u13 (0, lambda, 1) else (0, 1, lambda)  
            - lambda: Component of n vector. made from a, b, c 
            - a, b, c: Coefficient of distance function.
    """
    # Get rotation matrix which convert C2 to C1
    Rr = R1.T @ R2 # R1 = WC1, R2 = WC2 -> C1C2 = C1W @ WC2 = R1.T @ R2
    
    # Get Rb matrix
    e = C2 - C1
    x = np.array([1, 0, 0], dtype=np.float32)
    Rb_vec = np.cross(x, e)
    Rb = R.from_rotvec(Rb_vec).as_matrix()

    # check whether two vector in egeomtry now
    emat = ematrix(Rr, e)
    if is_in_egeometry(u1, u2, emat): raise Exception("Already in same Epipolar geometry")


    # Convert u1, u2 from world coord to new coord
    u1 = Rb @ u1
    u2 = Rb @ Rr @ u2

    # extract component of unit vectors
    u12, u13 = u1[1], u1[2]
    u22, u23 = u2[1], u2[2]

    # calculate d
    d = u13 * u13 - u23 * u23 if np.abs(u12) > np.abs(u13) else u12 * u12 - u22 * u22
    # if np.abs(d) <= 1e-5: raise Exception("No validate triangulate points")
    
    # calculate lambda value
    lamb1 = (u12 * u13 - u22 * u23) - (u22 * u13 - u12 * u23)
    lamb1 /= d

    lamb2 = (u12 * u13 - u22 * u23) + (u22 * u13 - u12 * u23)
    lamb2 /= d

    s1_denom = np.sqrt(1 + lamb1 * lamb1)
    s2_denom = np.sqrt(1 + lamb2 * lamb2)

    # check who is the smallest value for formula (16) in paper
    s1 = np.abs(u12 + lamb1 * u13) + np.abs(u22 + lamb1 * u23)
    s1 /= s1_denom

    s2 = np.abs(u12 + lamb2 * u13) + np.abs(u22 + lamb2 * u23)
    s2 /= s2_denom

    # determine lamb
    lamb = lamb1 if s1 > s2 else lamb2

    # calculate n vector
    n = np.array([0, lamb, 1], dtype=np.float32) if np.abs(u12) > np.abs(u13) else np.array([0, 1, lamb], dtype=np.float32)

    n_norm = np.linalg.norm(n)
    nn = n_norm * n_norm

    # project unit vectors in sphere to epipolar geometry
    u1 = u1 - (u1.T @ n) / nn * n
    u2 = u2 - (u2.T @ n) / nn * n

    e_mat = np.array([
        [0, e[2], -e[1]],
        [-e[2], 0, e[0]],
        [e[1], -e[0], 0]
    ], dtype=np.float32)
    E = Rr @ e_mat

    # if np.abs(u1 @ E @ u2) > 1e-7: raise Exception("Not in Epipolar constraint")

    # Calculate point in C1 coord
    point = midpoint(u1, u2, C2 = e)

    # transform point in origin coord
    point = Rb.T @ point + C1
    
    return point


if __name__ == '__main__':
    # define vectors in world coord
    u1w = np.array([1.,2, 0.1])
    u2w = np.array([-1.,2, 0.])
    
    std = 0.05
    # u1w /= np.linalg.norm(u1w)
    # u2w /= np.linalg.norm(u2w)

    u1w += std * np.random.normal()
    u2w += std * np.random.normal()

    # define camera position in world coord
    C1 = np.array([0, 0, 0])
    C2 = np.array([2, 0, 0])

    # define camera orientation in world coord
    deg1, deg2 = np.deg2rad(20), np.deg2rad(30)
    R1_vec = np.array([1, 2, 1])
    R1_vec = deg1 * R1_vec / np.linalg.norm(R1_vec)
    R2_vec = np.array([1, 2.5, 1])
    R2_vec = deg1 * R2_vec / np.linalg.norm(R2_vec)

    diff_ang = np.arccos((R1_vec.T @ R2_vec) / (np.linalg.norm(R1_vec) * np.linalg.norm(R2_vec)))
    print(np.rad2deg(diff_ang))

    # R.T is rotating matrix. It convert vector from world to unit sphere
    R1 = R.from_rotvec(R1_vec).as_matrix()  
    R2 = R.from_rotvec(R2_vec).as_matrix()
    
    # convert vectors from world coord to unit sphere
    u1s = R1.T @ u1w
    u2s = R2.T @ u2w

    # When we use mid point method, we should use unit vectors in world coord
    mid_x = midpoint(u1w, u2w, C1, C2)

    # when we use sphere linear method, we should use unit vectors in unit sphere
    us = np.vstack((u1s, u2s))
    Cs = np.vstack((C1, C2)).reshape(2, -1)
    Rs = np.vstack((R1, R2)).reshape(2, 3, -1)
    lin_x = sph_lin(us, Rs, Cs)

    # calc easy values
    quad_x = sph_quad(u1=u1w, u2=u2w, R1=R1, R2=R2, C1=C1, C2=C2)
    abs_x = sph_abs(u1=u1w, u2=u2w, R1=R1, R2=R2, C1=C1, C2=C2)

    print(f"Mid point value: {mid_x}")
    print(f"Sph-Lin value: {lin_x}")
    print(f"Sph-Quad value: {quad_x}")
    print(f"Sph-Abs value: {abs_x}")