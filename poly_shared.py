"""
Openmm / Polychrom conventions for bonds, angles, and dihedrals
---------------------------------------------------------------

Bonds are defined by two particles (i, j) and have a rest length ℓ₀ and a spring constant Δℓ_kT.
Angles are defined by three particles (i, j, k) and have a rest angle θ₀ and a spring constant k_ang.
Angles live on (0, π) and the angle to keep consecutive monomers straight is π.
Dihedrals are defined by four particles (i, j, k, l) and have a rest angle φ₀ and a spring constant k_dihedral.
Dihedrals live on (-π, π) and the angle to keep 4 particles in a square planar configuration is 0.

Default bond length is 1 (unitless). All k are measured in kT/rad (angles/dihedrals).
Bonds are different - parametrized by Δℓ_kT - which is bond extension at which the energy is kT.

Dihedrals are usually used to keep a square planar. A typical configuration would be:
for a square made of particles clockwise: [i,j,k,l]: a dihedral (i,j,l,k,) for bending across the jl diagonal,
and (j,i,k,l) for bending across the ik diagonal. Both with angle of 0, and k_dihedral of 50-ish.

It takes a few hours, a 6-pack of beer, and a lot of bashing your head against the wall to understand
what the "dihedral angle" actually is, and to use it freely. ChatGPT gets it wrong 50% of the time too. Be careful.
"""

import numpy as np
import openmm as mm
from dataclasses import dataclass, field
from math import pi
from polychrom import forces
import polychrom.simulation


import math
from typing import Dict, List, OrderedDict, Tuple, Sequence

import numpy as np

# ============================================================
# Classes and data structures for molecule structure
# ============================================================

Bond = Tuple[int, int, float, float]  # (i, j, ℓ₀, Δℓ_kT)
Angle = Tuple[int, int, int, float, float]  # (i, j, k, θ₀, k_ang)
Dihedral = Tuple[int, int, int, int, float, float]  # (i, j, k, l, φ₀, k_dihedral)


@dataclass
class MoleculeStructure:
    n_rows: int  # number of rows in the ribbon
    start_particle_index: int  # index of the first particle in the ribbon
    positions: np.ndarray  # (N, 3) array of particle positions
    bonds: List[Bond] = field(default_factory=list)  # list of bond tuples (i, j, ℓ₀, Δℓ_kT)
    angles: List[Angle] = field(default_factory=list)  # list of angle tuples (i, j, k, θ₀, k_ang)
    dihedrals: List[Dihedral] = field(default_factory=list)  # list of dihedral tuples (i, j, k, l, φ₀, k_dihedral)
    index_dict: Dict[Tuple[int, int], int] = field(default_factory=dict)  # look up global index by (layer, vertex_id)

    def get_next_idx(self) -> int:
        """
        Returns the index of the next particle after the last one in this structure.
        Useful for concatenating multiple structures.
        """
        return self.start_particle_index + len(self.positions)

    def get_end_idx(self) -> int:
        """
        Returns the index of the last particle in this structure.
        Useful for concatenating multiple structures.
        """
        return self.start_particle_index + len(self.positions) - 1

    def _pos(self, idx):
        offset = self.start_particle_index
        return self.positions[idx - offset]


@dataclass
class MoleculeStructureCollection:
    structures: OrderedDict[str, MoleculeStructure] = field(default_factory=OrderedDict)

    def __getitem__(self, name: str) -> MoleculeStructure:
        return self.structures[name]

    def __setitem__(self, name: str, structure: MoleculeStructure) -> None:
        self.structures[name] = structure

    def __iter__(self):
        return iter(self.structures)

    def __len__(self) -> int:
        return len(self.structures)

    def keys(self):
        return self.structures.keys()

    def values(self):
        return self.structures.values()

    def items(self):
        return self.structures.items()

    def next_idx(self) -> int:
        if not self.structures:
            return 0
        return next(reversed(self.structures.values())).get_next_idx()


@dataclass
class MoleculeStructureWithNames:
    n_row_dict: dict[str, int]  # number of rows in each substructure
    start_particle_dict: dict[str, int]  # index of the first particle in each substructure
    positions: np.ndarray  # (N, 3) array of particle positions
    bonds: List[Bond]  # list of bond tuples (i, j, ℓ₀, Δℓ_kT)
    angles: List[Angle]  # list of angle tuples (i, j, k, θ₀, k_ang)
    dihedrals: List[Dihedral]  # list of dihedral tuples (i, j, k, l, φ₀, k_dihedral)
    index_dict: Dict[Tuple[str, int, int], int]  # look up global index by (name, layer, vertex_id)


# =============================================================================
# Helper functions for verifying and concatenating molecule structures
# =============================================================================


def verify_bond_angle_dihedral_uniqeness(mol: MoleculeStructure | MoleculeStructureWithNames) -> None:
    """
    Verifies that bonds, angles, and dihedrals in a MoleculeStructure are unique.
    Takes into account the fact that bonds, angles, and dihedrals are "mirroring invariant",
    i.e. (i,j) is the same as (j,i) for bonds, and (i,j,k) is the same as (k,j,i) for angles, etc.

    Raises ValueError if any duplicates are found, indicating the duplicated record (eager is fine).
    """
    bond_check_dict = {}
    for ind, bond in enumerate(mol.bonds):
        key = tuple(sorted((bond[0], bond[1])))
        if key in bond_check_dict:
            offending_ind, offending_bond = bond_check_dict[key]
            raise ValueError(f"Duplicate bond found: {bond} at position {ind} and {offending_bond} at {offending_ind}.")
        bond_check_dict[key] = (ind, bond)

    angle_check_dict = {}
    for ind, angle in enumerate(mol.angles):
        if angle[0] == angle[1] or angle[1] == angle[2] or angle[0] == angle[2]:
            raise ValueError(f"Invalid angle with repeated indices found: {angle} at position {ind}.")
        if angle[0] > angle[2]:
            key = (angle[2], angle[1], angle[0])
        else:
            key = (angle[0], angle[1], angle[2])
        if key in angle_check_dict:
            offending_ind, offending_angle = angle_check_dict[key]
            raise ValueError(
                f"Duplicate angle found: {angle} at position {ind} and {offending_angle} at {offending_ind}."
            )
        angle_check_dict[key] = (ind, angle)

    dihedral_check_dict = {}
    for ind, dihedral in enumerate(mol.dihedrals):
        if len(set(dihedral[:4])) < 4:
            raise ValueError(f"Invalid dihedral with repeated indices found: {dihedral} at position {ind}.")
        if dihedral[0] > dihedral[3]:
            key = (dihedral[3], dihedral[2], dihedral[1], dihedral[0])
        else:
            key = (dihedral[0], dihedral[1], dihedral[2], dihedral[3])
        if key in dihedral_check_dict:
            offending_ind, offending_dihedral = dihedral_check_dict[key]
            raise ValueError(
                f"Duplicate dihedral found: {dihedral} at position {ind} and {offending_dihedral} at {offending_ind}."
            )
        dihedral_check_dict[key] = (ind, dihedral)


def concatenate_molecule_structures(
    structureDict: OrderedDict[str, MoleculeStructure] | MoleculeStructureCollection,
    extra_bonds: List[Bond] | None = None,
    extra_angles: List[Angle] | None = None,
    extra_dihedrals: List[Dihedral] | None = None,
) -> MoleculeStructureWithNames:
    """
    Concatenate multiple MoleculeStructure dictionaries into one.
    Ensures indices don't overlap and are consecutive.
    Indices should be already linear and unique across all structures.

    Parameters
    ----------
    structures : List[MoleculeStructure]
        List of MoleculeStructure dictionaries to concatenate.
    names : List[str] | None, optional
        List of names corresponding to each structure, used for index lookup
    Returns
    -------
    MoleculeStructure
        A single MoleculeStructure containing concatenated positions, bonds, angles, and index_dict.
    """
    if isinstance(structureDict, MoleculeStructureCollection):
        structureDict = structureDict.structures

    # making sure indices are unique and consecutive
    count_array = np.zeros(sum(len(s.positions) for s in structureDict.values()), dtype=int)
    for s in structureDict.values():
        verify_bond_angle_dihedral_uniqeness(s)
        for idx in s.index_dict.values():
            count_array[idx] += 1
    if np.any(count_array > 1):
        raise ValueError("Indices in structures are not unique. Please ensure each structure has unique indices.")
    if np.any(count_array < 1):
        raise ValueError("Indices in structures are not consecutive. Please ensure indices are consecutive.")

    total_positions = np.concatenate([s.positions for s in structureDict.values()], axis=0)
    total_bonds = [bond for s in structureDict.values() for bond in s.bonds]
    total_angles = [angle for s in structureDict.values() for angle in s.angles]
    total_dihedrals = [dihedral for s in structureDict.values() for dihedral in s.dihedrals]

    # Update index_dict to account for global indices
    index_dict = {}
    for name, s in structureDict.items():
        for (layer, vid), idx in s.index_dict.items():
            index_dict[(name, layer, vid)] = idx

    if extra_bonds is not None:
        total_bonds.extend(extra_bonds)
    if extra_angles is not None:
        total_angles.extend(extra_angles)
    if extra_dihedrals is not None:
        total_dihedrals.extend(extra_dihedrals)

    # Create the combined structure
    structure = MoleculeStructureWithNames(
        n_row_dict={name: s.n_rows for name, s in structureDict.items()},
        start_particle_dict={name: s.start_particle_index for name, s in structureDict.items()},
        positions=total_positions,
        bonds=total_bonds,
        angles=total_angles,
        dihedrals=total_dihedrals,
        index_dict=index_dict,
    )

    verify_bond_angle_dihedral_uniqeness(structure)

    return structure


# =============================================================================
# Helper function to add forces to a simulation
# =============================================================================


# Create and add forces
def add_polymer_forces(
    sim, bonds: list[Bond], angles: list[Angle], dihedrals: list[Dihedral]
) -> tuple[dict[tuple[int, int, int, int], int], mm.CustomTorsionForce] | None:
    """
    Add harmonic bonds, angle forces, and dihedral forces to a simulation.

    Parameters
    ----------
    sim : polychrom.simulation.Simulation
        The simulation object to add forces to
    bonds : List[Bond]
        List of bond tuples (i, j, ℓ₀, Δℓ_kT)
    angles : List[Angle]
        List of angle tuples (i, j, k, θ₀, k_ang)
    dihedrals : List[Dihedral]
        List of dihedral tuples (i, j, k, l, φ₀, k_dihedral)

    Returns
    -------
    None
    """
    # Process bonds
    bond_pos = [(i[0], i[1]) for i in bonds]  # convert to positions
    bond_lengths = [float(i[2]) for i in bonds]  # convert to lengths
    bond_devs = [float(i[3]) for i in bonds]  # convert to widths
    bond_force = forces.harmonic_bonds(sim, bond_pos, bondLength=bond_lengths, bondWiggleDistance=bond_devs)
    sim.add_force(bond_force)

    # Process angles
    if angles:
        angle_pos = [(i[0], i[1], i[2]) for i in angles]  # convert to angle force triplets
        angle_thetas = [float(i[3]) for i in angles]  # convert to angle force default angle
        angle_ks = [float(i[4]) for i in angles]  # convert angle force spring constants
        angle_force = forces.angle_force(sim, angle_pos, theta_0=angle_thetas, k=angle_ks)
        sim.add_force(angle_force)

    # Process dihedrals
    if dihedrals:
        dihedral_force = mm.CustomTorsionForce("k * kT * (1 - cos(theta - theta_0))")
        dihedral_force.name = "dihedral_force"  # type: ignore - this is polychrom's old convention
        dihedral_force.addPerTorsionParameter("theta_0")
        dihedral_force.addPerTorsionParameter("k")
        dihedral_force.addGlobalParameter("kT", sim.kT)

        dihedral_pos = [(i[0], i[1], i[2], i[3]) for i in dihedrals]
        dihedral_thetas = [float(i[4]) for i in dihedrals]
        dihedral_ks = [float(i[5]) for i in dihedrals]

        DihedralIndDict = {}

        for pos, theta, k in zip(dihedral_pos, dihedral_thetas, dihedral_ks):
            record = (pos[0], pos[1], pos[2], pos[3], (theta, k))
            ind = dihedral_force.addTorsion(*record)
            DihedralIndDict[(pos[0], pos[1], pos[2], pos[3])] = ind

        sim.add_force(dihedral_force)
        return DihedralIndDict, dihedral_force
    return None


# ==================================================================
# Helper functions for computing angles and dihedrals, and distances
# ==================================================================


def dihedral_angle(positions: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """
    Compute the signed dihedral angle (in radians, range [-π, π]) defined by four points.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Array of 3D coordinates.
    i, j, k, l : int
        Indices of the four particles, in order.

    Returns
    -------
    float
        Dihedral angle between the plane (i,j,k) and the plane (j,k,l).
    """
    p1 = positions[i]
    p2 = positions[j]
    p3 = positions[k]
    p4 = positions[l]

    # Bond vectors
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normal vectors to the planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize b2 for the sign calculation
    b2_unit = b2 / np.linalg.norm(b2)

    # Components for atan2
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), b2_unit)

    return np.arctan2(y, x)


def find_minimum_triplet_distance(
    pos: np.ndarray,
    type1_indices: list[int],
    type2_indices: list[int],
    type3_indices: list[int],
    cutoff: float = 2.0,
) -> Tuple[float, np.ndarray, int]:
    """
    Find the minimum pairwise distance across all possible triplets and return the indices of the triplet with the minimum distance.
    Also returns the number of triplets with distance < cutoff

    Parameters:
    -----------
    pos : array-like, shape (N, 3)
        Positions of all particles
    type1_indices : array-like
        Indices of type 1 particles
    type2_indices : array-like
        Indices of type 2 particles
    type3_indices : array-like
        Indices of type 3 particles

    Returns:
    --------
    min_distance : float
        The minimum distance found
    best_triplet : np.ndarray
        Indices (i1, i2, i3) of the triplet with minimum distance
    """

    # Get positions for each type
    pos1 = pos[type1_indices]  # shape: (n_type1, 3)
    pos2 = pos[type2_indices]  # shape: (n_type2, 3)
    pos3 = pos[type3_indices]  # shape: (n_type3, 3)

    min_distance = np.inf
    best_triplet = np.array((1, 1, 1))  # Initialize with dummy values

    num_triplets = 0  # Counter for triplets with distance < cutoff
    # Loop over type1 and type2 (small arrays)
    for i1, idx1 in enumerate(type1_indices):
        for i2, idx2 in enumerate(type2_indices):
            # Vectorized distance calculation for all type3
            d12 = np.linalg.norm(pos1[i1] - pos2[i2])
            d13 = np.linalg.norm(pos1[i1] - pos3, axis=1)  # vectorized over type3
            d23 = np.linalg.norm(pos2[i2] - pos3, axis=1)  # vectorized over type3

            # Minimum distance for each type3 particle with this (type1, type2) pair
            triplet_min_distances = np.maximum(d12, np.maximum(d13, d23))

            # Find the best type3 for this (type1, type2) pair
            best_i3 = np.argmin(triplet_min_distances)
            current_min = triplet_min_distances[best_i3]
            if current_min < cutoff:
                num_triplets += (triplet_min_distances < cutoff).sum()

            if current_min < min_distance:
                min_distance = current_min
                best_triplet = np.array((idx1, idx2, type3_indices[best_i3]))

    return float(min_distance), best_triplet, num_triplets


# =============================================================================
# Functions to build various components of a molecular structure
# =============================================================================


def build_ribbon(
    n_rows: int,
    start_particle_index: int = 0,
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    bond_dl: float = 0.05,
    right_angle_k: float = 100.0,
    backbone_angle_k: float = 100.0,
    dihedral_k: float = 80.0,
    dihedral_offset: float = 1,
    break_idx: int | None = None,
    break_sign: float = 1.0,
) -> MoleculeStructure:
    """
    Build a width-2 torsion-resistant ribbon. It is stabilized by:
    * Harmonic bonds between adjacent particles.
    * Four right angles for each "square" in the ribbon.
    * One backbone angle for each monomer not at the end to keep the backbone straight.
    * Two dihedrals for each square (folding across the two diagonals).

    Initial conformation is a flat ribbon growing in +Z. Ribbon is flat along X.
    All neighbor distances = 1 (unitless).

    at break_idx -3 to break_idx - 2 we put a "torque sensing" unit:
    * We disable bond along vind=1
    * Disable backbone angle along vind=1 at break_ind-3 and break_ind-2
    * Disable dihedrals across the two diagonals at break_ind-3 and break_ind-2.
    * Disable right angles involving vind=1 at break_ind-3 and break_ind-2.
    """
    if break_idx is None:
        break_idx = -1000  # no break point
    ts_idx = break_idx - 3  # index of the first particle in the torque sensing unit
    # total particles = 2 per row
    total = n_rows * 2
    positions = np.zeros((total, 3), dtype=float)
    bonds: List[Bond] = []
    angles: List[Angle] = []
    dihedrals: List[Dihedral] = []
    index_dict: Dict[Tuple[int, int], int] = {}

    # Define twist per row (radians)
    # empirically found to be "OK" at initialization and
    # it is basically forgotten after a few blocks, as forces define the structure.
    twist_per_row = -0.43 * dihedral_offset

    # Assign positions and index_dict
    for row in range(n_rows):
        z = start[2] + row * 1.0
        # Calculate twist angle for this row (no twist for the last row)
        twist_angle = (n_rows - 1 - row) * twist_per_row

        for col in (0, 1):
            idx = start_particle_index + row * 2 + col
            # Base position before twist
            x_base = start[0] + col * 1.0
            y_base = start[1]

            # Apply twist rotation in XY plane
            x_center = start[0] + 0.5  # Center of the ribbon in X
            y_center = start[1]
            x = x_center + (x_base - x_center) * np.cos(twist_angle)  # - (y_base - y_center) * np.sin(twist_angle)
            y = y_center + (x_base - x_center) * np.sin(twist_angle)  # + (y_base - y_center) * np.cos(twist_angle)

            positions[idx - start_particle_index] = (x, y, z)
            index_dict[(row, col)] = idx

    # Bonds: horizontal + vertical
    for row in range(n_rows):
        left = index_dict[(row, 0)]
        right = index_dict[(row, 1)]
        # horizontal bond
        bonds.append((left, right, 1.0, bond_dl))
        # vertical bonds (except last row)
        if row < n_rows - 1:
            up_left = index_dict[(row + 1, 0)]
            up_right = index_dict[(row + 1, 1)]
            bonds.append((left, up_left, 1.0, bond_dl))
            if row != ts_idx:  # skip vertical bonds at the torque sensing unit
                bonds.append((right, up_right, 1.0, bond_dl))

    # Right-angle constraints for each square
    for row in range(n_rows - 1):
        if row == break_idx or row == break_idx - 1:
            continue
        i_idx = index_dict[(row, 0)]
        j_idx = index_dict[(row, 1)]
        k_idx = index_dict[(row + 1, 1)]
        l_idx = index_dict[(row + 1, 0)]
        # bottom-left corner vind=(1,0,0)
        angles.append((j_idx, i_idx, l_idx, pi / 2.0, right_angle_k))
        # top-left corner vind=(0,0,1)
        angles.append((i_idx, l_idx, k_idx, pi / 2.0, right_angle_k))
        if row != ts_idx:  # skip right angles at the torque sensing unit
            # bottom-right corner vind=(0,1,1)
            angles.append((i_idx, j_idx, k_idx, pi / 2.0, right_angle_k))
            # top-right corner vind=(1,1,0)
            angles.append((j_idx, k_idx, l_idx, pi / 2.0, right_angle_k))

    # Backbone straightness (angles along each chain)
    for row in range(1, n_rows - 1):
        if row == break_idx or row + 1 == break_idx or row - 1 == break_idx:
            continue
        for col in (0, 1):
            if (row == ts_idx or row == ts_idx + 1) and col == 1:
                continue  # skip the torque sensing unit
            prev_idx = index_dict[(row - 1, col)]
            curr_idx = index_dict[(row, col)]
            next_idx = index_dict[(row + 1, col)]
            angles.append((prev_idx, curr_idx, next_idx, pi, backbone_angle_k))

    # Dihedrals: two per square (across both diagonals)
    for row in range(n_rows - 1):
        # if row == break_idx or row + 1 == break_idx:  # allow flexibility at the break point
        #     continue
        if row == ts_idx:  # this row will be absorbing twist
            continue

        no_offset_rows = [0, n_rows - 2]  # first and last "square" do not have dihedral offsets
        if break_idx:  # same for the squares around the break point
            no_offset_rows.extend(list(range(break_idx - 3, break_idx + 3)))

        do = pi if row in no_offset_rows else pi + dihedral_offset

        i_idx = index_dict[(row, 0)]
        j_idx = index_dict[(row, 1)]
        k_idx = index_dict[(row + 1, 1)]
        l_idx = index_dict[(row + 1, 0)]
        # It was experimentally verified that offsets are the same sign for both diagonals.
        # diagonal j–l
        dihedrals.append((i_idx, j_idx, l_idx, k_idx, do, dihedral_k))
        # diagonal i–k
        dihedrals.append((j_idx, i_idx, k_idx, l_idx, do, dihedral_k))

    # creating a 120-degree angle before/after the ribbon break
    if break_idx > 0:
        for row, angle in ((break_idx - 1, -2 * pi / 3), (break_idx + 1, -2 * pi / 3)):
            a_idx = index_dict[(row - 1, 0)]
            b_idx = index_dict[(row - 1, 1)]
            c_idx = index_dict[(row, 0)]
            d_idx = index_dict[(row, 1)]
            e_idx = index_dict[(row + 1, 0)]
            f_idx = index_dict[(row + 1, 1)]
            dihedrals.append((a_idx, c_idx, d_idx, f_idx, angle * break_sign, 2 * dihedral_k))
            dihedrals.append((b_idx, c_idx, d_idx, e_idx, angle * break_sign, 2 * dihedral_k))

    return MoleculeStructure(
        n_rows=n_rows,
        start_particle_index=start_particle_index,
        positions=positions,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        index_dict=index_dict,
    )


def connect_ribbon_ends(
    ribbon1: MoleculeStructure,
    ribbon2: MoleculeStructure,
    extra_bonds: List[Bond],
    extra_angles: List[Angle],
    extra_dihedrals: List[Dihedral],
    bond_dl: float = 0.05,
    angle_k: float = 50.0,
    dihedral_k: float = 50.0,
    last_row_length: float = 2.5,
    penult_row_length: float = 2.3,
) -> None:
    """
    Connect two parallel width-2 ribbons only at their "far" end (rows n-2 and n-1),
    forming a single rigid "cube" at that end.  Updates the extra_bonds, extra_angles,
    extra_dihedrals lists in place.

    Parameters
    ----------
    ribbon1, ribbon2 : MoleculeStructure
        Two ribbon structures with the same n_rows.
    extra_bonds : list to append new Bond tuples
    extra_angles : list to append new Angle tuples
    extra_dihedrals : list to append new Dihedral tuples
    bond_dl : float
        Δℓ_kT for all new bonds.
    angle_k : float
        k_ang for all new right‐angle constraints.
    dihedral_k : float
        k_dihedral for all new planar dihedral restraints.
    """
    n1 = ribbon1.n_rows
    n2 = ribbon2.n_rows
    if n1 != n2:
        raise ValueError("Both ribbons must have the same n_rows")

    # rows to join
    penult = n1 - 2
    last = n1 - 1

    # fetch the four corner indices of each of the two rows
    A0 = ribbon1.index_dict[(penult, 0)]
    A1 = ribbon1.index_dict[(penult, 1)]
    B0 = ribbon1.index_dict[(last, 0)]
    B1 = ribbon1.index_dict[(last, 1)]

    C0 = ribbon2.index_dict[(penult, 0)]
    C1 = ribbon2.index_dict[(penult, 1)]
    D0 = ribbon2.index_dict[(last, 0)]
    D1 = ribbon2.index_dict[(last, 1)]

    # 1) Bonds across the two end‐rows, for each column
    for i, j in ((A0, C0), (A1, C1)):
        extra_bonds.append((i, j, penult_row_length, bond_dl))

    for i, j in ((B0, D0), (B1, D1)):
        extra_bonds.append((i, j, last_row_length, bond_dl))

    # 2) Define the four new square faces at that end
    faces = [
        [A0, A1, C1, C0],  # top face (penultimate row)
        [B0, B1, D1, D0],  # bottom face (last row)
        [A0, C0, D0, B0],  # left face (column 0)
        [A1, C1, D1, B1],  # right face (column 1)
    ]

    for quad in faces:
        i, j, k, l = quad
        # a) Four right‐angle constraints at each corner
        extra_angles += [
            (j, i, l, pi / 2, angle_k),
            (k, j, i, pi / 2, angle_k),
            (l, k, j, pi / 2, angle_k),
            (i, l, k, pi / 2, angle_k),
        ]
        # b) Two planar (improper) dihedrals across the two diagonals
        extra_dihedrals += [
            (i, j, l, k, pi, dihedral_k),
            (j, i, k, l, pi, dihedral_k),
        ]


def connect_with_polymer(
    struct1: MoleculeStructure,
    idx1: int,
    struct2: MoleculeStructure,
    idx2: int,
    start_mon_idx: int,
    n_monomers: int,
    extra_bonds: List[Bond],
    bond_dl: float = 0.05,
) -> MoleculeStructure:
    """
    Build a simple zig-zag polymer of n_monomers between particle idx1 in struct1
    and idx2 in struct2.  The new monomers are assigned indices
    start_mon_idx .. start_mon_idx + n_monomers - 1.  Bonds between consecutive
    monomers have rest length = total_dist / (n_monomers + 1), and the chain
    zig-zags in the plane perpendicular to the connection vector with amplitude
    zigzag_amplitude.  Updates extra_bonds in place to attach the first monomer
    to idx1 and the last monomer to idx2.

    Returns a MoleculeStructure for just the polymer chain (positions + internal bonds).
    """
    # Fetch endpoint positions

    v = struct2._pos(idx2) - struct1._pos(idx1)
    L = np.linalg.norm(v)
    if L == 0:
        raise ValueError("Endpoints coincide; cannot build polymer.")
    u = v / L
    if np.linalg.norm(u) > 1:
        raise ValueError("Endpoints are too far apart; cannot build polymer.")

    # calculate zigzag amplitude to keep bond length at 1
    zigzag_amplitude = math.sqrt(1 - (L / (n_monomers + 1)) ** 2) / 2

    # build a perpendicular basis vector
    # pick any vector not colinear with u
    if abs(u[0]) < 0.9:
        arb = np.array([1.0, 0.0, 0.0])
    else:
        arb = np.array([0.0, 1.0, 0.0])
    # p = normalized cross(u, arb)
    p = np.cross(u, arb)
    p /= np.linalg.norm(p)

    # allocate positions
    positions = np.zeros((n_monomers, 3), dtype=float)
    index_dict = {}
    bonds: List[Bond] = []

    for i in range(n_monomers):
        t = (i + 1) / (n_monomers + 1)
        base = struct1._pos(idx1) + u * (L * t)
        # alternate zigzag sign
        offset = ((-1) ** i) * zigzag_amplitude * p
        positions[i] = base + offset
        index = start_mon_idx + i
        index_dict[(i, 0)] = index

    # internal polymer bonds
    for i in range(n_monomers - 1):
        i_idx = start_mon_idx + i
        j_idx = start_mon_idx + i + 1
        bonds.append((i_idx, j_idx, 1, bond_dl))

    # connect first monomer to idx1
    first_idx = start_mon_idx
    extra_bonds.append((idx1, first_idx, 1, bond_dl))
    # connect last monomer to idx2
    last_idx = start_mon_idx + n_monomers - 1
    extra_bonds.append((last_idx, idx2, 1, bond_dl))

    return MoleculeStructure(
        n_rows=n_monomers,
        start_particle_index=start_mon_idx,
        positions=positions,
        bonds=bonds,
        angles=[],
        dihedrals=[],
        index_dict=index_dict,
    )


def build_hairy_ring(
    radius: float,
    n_monomers: int,
    p_hairy: float = 0.1,
    seed: int | None = None,
    backbone_k: float = 1.5,
    step: float = 0.9,
    offset_scale: float = 0.05,
    start_mon_idx: int = 0,
    plane: str = "XY",
) -> MoleculeStructure:
    """
    Build a “hairy” polymer ring in the XY plane of given radius.
    index_dict maps (row, vind) -> global index, where vind=0 is backbone
    and vind=1 is the single hair at that row (if present).

    Parameters
    ----------
    radius        : ring radius
    n_monomers    : number of backbone monomers
    p_hairy       : fraction of backbone monomers with one “hair”
    seed          : RNG seed for reproducibility
    backbone_k    : angle stiffness in kT/rad^2
    step          : backbone bond length
    offset_scale  : RMS positional jitter
    start_mon_idx : global index of the first monomer
    """
    rng = np.random.default_rng(seed)
    # decide which rows get hairs
    n_hairy = int(p_hairy * n_monomers)
    hair_rows = set(rng.choice(n_monomers, n_hairy, replace=False))

    # total particles = backbone + hairy
    total = n_monomers + len(hair_rows)
    positions = np.zeros((total, 3), float)
    struct = MoleculeStructure(n_monomers, start_mon_idx, positions.copy())

    # build backbone
    for i in range(n_monomers):
        θ = 2 * math.pi * i / n_monomers
        base = np.array([radius * math.cos(θ), radius * math.sin(θ), 0.0])
        pos = base + rng.normal(scale=offset_scale, size=3)
        gi = start_mon_idx + i
        struct.positions[i] = pos
        struct.index_dict[(i, 0)] = gi

    coord1 = struct.positions[:, 0].copy()  # X coordinates
    coord2 = struct.positions[:, 1].copy()  # Y coordinates
    if plane == "XZ":
        struct.positions[:, 0] = coord1  # X stays X
        struct.positions[:, 1] = 0  # Y becomes zero
        struct.positions[:, 2] = coord2  # Z is now Y
    elif plane == "YZ":
        struct.positions[:, 0] = 0  # X coordinates becomes zero
        struct.positions[:, 1] = coord1  # Y is now Z
        struct.positions[:, 2] = coord2  # Z is now Y

    # backbone bonds & angles
    for i in range(n_monomers):
        j = (i + 1) % n_monomers
        struct.bonds.append((start_mon_idx + i, start_mon_idx + j, 1, offset_scale))
        prev = (i - 1) % n_monomers
        struct.angles.append((start_mon_idx + prev, start_mon_idx + i, start_mon_idx + j, math.pi, backbone_k))

    # build hairs
    next_idx = start_mon_idx + n_monomers
    for i in range(n_monomers):
        if i not in hair_rows:
            continue
        base_pos = struct.positions[i]
        # random unit vector
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        hair_pos = base_pos + step * v + rng.normal(scale=offset_scale, size=3)
        struct.positions[next_idx - start_mon_idx] = hair_pos
        struct.bonds.append((start_mon_idx + i, next_idx, 1, offset_scale))
        struct.index_dict[(i, 1)] = next_idx
        next_idx += 1

    return struct


def build_straight_line_through_opening(
    n_monomers: int,
    spacing: float = 0.95,
    start_mon_idx: int = 0,
    backbone_k: float = 5.0,
    offset_scale: float = 0.05,
    seed: int | None = None,
    p_hairy: float = 0.1,
    hair_len: int = 2,
) -> MoleculeStructure:
    """
    Build a straight polymer line that passes through the opening between two ribbons.

    The line runs along the X axis, passing through (1.5, 0, 1.5).
    Optionally adds "hairs" (short side chains) at density p_hairy, each hair being
    `hair_len` connected monomers with bonds only (no angles/dihedrals on hairs).
    """
    rng = np.random.default_rng(seed)

    # decide which backbone monomers get hairs (same convention as build_hairy_ring)
    n_hairy = int(p_hairy * n_monomers)
    hair_rows = set(rng.choice(n_monomers, n_hairy, replace=False)) if n_hairy > 0 else set()

    # total particles = backbone + sum hair lengths
    total = n_monomers + len(hair_rows) * hair_len
    positions = np.zeros((total, 3), dtype=float)
    index_dict: Dict[Tuple[int, int], int] = {}
    bonds: List[Bond] = []
    angles: List[Angle] = []

    center = np.array([1.5, 0.0, 1.5])
    total_length = (n_monomers - 1) * spacing
    x_start = center[0] - total_length / 2

    # backbone positions + index_dict
    for i in range(n_monomers):
        x = x_start + i * spacing
        pos = np.array([x, center[1], center[2]], dtype=float)
        pos += rng.normal(scale=offset_scale, size=3)
        positions[i] = pos
        index_dict[(i, 0)] = start_mon_idx + i

    # backbone bonds
    for i in range(n_monomers - 1):
        bonds.append((start_mon_idx + i, start_mon_idx + i + 1, spacing, offset_scale))

    # backbone angle constraints to keep the line straight
    for i in range(1, n_monomers - 1):
        angles.append((start_mon_idx + i - 1, start_mon_idx + i, start_mon_idx + i + 1, math.pi, backbone_k))

    # hairs: appended after backbone in positions array
    next_idx = start_mon_idx + n_monomers
    next_pos = n_monomers  # positions[] cursor for hair particles

    # choose a roughly perpendicular direction (biased toward +Y/+Z so it doesn't sit on the line)
    base_dir = np.array([0.0, 1.0, 1.0], dtype=float)
    base_dir /= np.linalg.norm(base_dir)

    for i in range(n_monomers):
        if i not in hair_rows or hair_len <= 0:
            continue

        prev_global = start_mon_idx + i
        prev_pos = positions[i]

        for h in range(hair_len):
            # small random tilt so hairs don't all overlap
            v = base_dir + 0.25 * rng.normal(size=3)
            v /= np.linalg.norm(v)

            hair_pos = prev_pos + spacing * v + rng.normal(scale=offset_scale, size=3)
            positions[next_pos] = hair_pos

            gidx = next_idx
            index_dict[(i, 1 + h)] = gidx

            bonds.append((prev_global, gidx, spacing, offset_scale))

            prev_global = gidx
            prev_pos = hair_pos
            next_idx += 1
            next_pos += 1

    return MoleculeStructure(
        n_rows=n_monomers,
        start_particle_index=start_mon_idx,
        positions=positions,
        bonds=bonds,
        angles=angles,
        dihedrals=[],
        index_dict=index_dict,
    )


def selective_attraction(
    sim_object: polychrom.simulation.Simulation,
    type_a_indices: Sequence[int],
    type_b_indices: Sequence[int],
    attraction_energy: float = 1.0,
    attraction_radius: float = 3.0,
    name: str = "selective_attraction",
) -> mm.CustomNonbondedForce:
    """
    Weak attractive interaction between two particle sets (A-B only, not A-A or B-B).

    Uses a smooth quartic well: E = -ε * (1 - (r/σ)²)² for r < σ
    Well depth ε at r=0, smoothly goes to 0 at r=σ with zero derivative at both ends.

    Parameters
    ----------
    type_a_indices : particles in set A (e.g., ribbon opening)
    type_b_indices : particles in set B (e.g., line)
    attraction_energy : well depth in kT
    attraction_radius : cutoff distance in sim units
    """
    energy = "-ATTRe * (1 - (r/ATTRsigma)^2)^2 * step(ATTRsigma - r)"

    force = mm.CustomNonbondedForce(energy)
    force.name = name  # type: ignore
    force.addGlobalParameter("ATTRe", attraction_energy * sim_object.kT)
    force.addGlobalParameter("ATTRsigma", attraction_radius * sim_object.conlen)

    force.setCutoffDistance(attraction_radius * sim_object.conlen)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)

    # All particles must be added, but no per-particle params needed
    for _ in range(sim_object.N):
        force.addParticle([])

    # Restrict to only A-B interactions
    force.addInteractionGroup(set(type_a_indices), set(type_b_indices))

    return force


def build_line_attached_to_monomer(
    struct: MoleculeStructure,
    idx_connect_to: int,
    n_monomers: int,
    start_mon_idx: int,
    extra_bonds: List[Bond],
    bond_dl: float = 0.05,
) -> MoleculeStructure:
    """
    Build a simple line of n_monomers attached to idx in struct.
    The new monomers are assigned indices start_mon_idx .. start_mon_idx + n_monomers - 1.
    Bonds between consecutive monomers have rest length = 1 and wiggle distance bond_dl.
    Returns a MoleculeStructure for just the line (positions + internal bonds).
    """
    positions = np.zeros((n_monomers, 3), dtype=float)
    index_dict = {}
    bonds: List[Bond] = []

    for i in range(n_monomers):
        positions[i] = struct._pos(idx_connect_to) + np.array([0.0, i + 0.7, i + 0.7])
        index_dict[(i, 0)] = start_mon_idx + i

    # internal polymer bonds
    for i in range(n_monomers - 1):
        i_idx = start_mon_idx + i
        j_idx = start_mon_idx + i + 1
        bonds.append((i_idx, j_idx, 1, bond_dl))

    # connect first monomer to idx
    extra_bonds.append((idx_connect_to, start_mon_idx, 1, bond_dl))

    return MoleculeStructure(
        n_rows=n_monomers,
        start_particle_index=start_mon_idx,
        positions=positions,
        bonds=bonds,
        angles=[],
        dihedrals=[],
        index_dict=index_dict,
    )


# =============================================================================
# OpenMM forces
# =============================================================================


def attractiveBondForce(
    sim_object: polychrom.simulation.Simulation,
    bonds: list[tuple[int, int]],
    strength_kt: float = 2,
    cutoff: float = 2.0,
    name: str = "attr_bonds",
) -> mm.CustomBondForce:
    """
    Attractive bond force that mimics the nonbonded potential.
    (it is based on the polynomial repulsive force, just negative of it)
    A workaround to enable a simple short-range attraction between just these two monomers.
    """

    energy = (
        "- step(ATTRsigma - r) * eattr;"
        "eattr = rsc12 * (rsc2 - 1.0) * ATTRe / emin12 + ATTRe;"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / ATTRsigma * rmin12;"
    )

    force = mm.CustomBondForce(energy)
    force.name = name  # type: ignore

    force.addGlobalParameter("ATTRsigma", sim_object.conlen * cutoff)
    force.addGlobalParameter("ATTRe", sim_object.kT * strength_kt)
    force.addGlobalParameter("emin12", 46656.0 / 823543.0)
    force.addGlobalParameter("rmin12", np.sqrt(6.0 / 7.0))

    for i, j in bonds:
        force.addBond(int(i), int(j), [])

    return force


# A key force to keep NipBL, hinge stick, and DNA together
def threeWayAttraction(
    sim_object: polychrom.simulation.Simulation,
    type1_particle_idx: Sequence[int],
    type2_particle_idx: Sequence[int],
    type3_particle_idx: Sequence[int],
    attractionEnergy: float = 0.5,
    attractionRadius: float = 2.0,
    name: str = "three_way_attraction",
) -> mm.CustomManyParticleForce:
    energy = (
        # first equation is the energy. Multiply it by ATTRe to give it "units"
        # (all the parts below were just between 0 and 1 in Y scale)
        # Energy is negative since this is the attraction potential
        "-pot_multiply * ATTRe;"  # ATTRe is a global parameter, energy is in kT
        # multiply all 3 potentials together - the product should go to zero if any two particles are at the cutoff
        "pot_multiply = pot12 * pot13 * pot23;"
        "pot13 = rsc13_12 * (rsc13_2 - 1.0) / emin12 + 1;"  # emin12 is a constant, 12 is power, not particle index
        "rsc13_12 = rsc13_4 * rsc13_4 * rsc13_4;"
        "rsc13_4 = rsc13_2 * rsc13_2;"
        "rsc13_2 = rsc13_1 * rsc13_1;"
        "rsc13_1 = r13 / REPsigma * rmin12;"  # rmin12 is a constant, 12 is power, not particle index
        "pot23 = rsc23_12 * (rsc23_2 - 1.0) / emin12 + 1;"  # emin12 is a constant, 12 is power, not particle index
        "rsc23_12 = rsc23_4 * rsc23_4 * rsc23_4;"
        "rsc23_4 = rsc23_2 * rsc23_2;"
        "rsc23_2 = rsc23_1 * rsc23_1;"
        "rsc23_1 = r23 / REPsigma * rmin12;"  # rmin12 is a constant, 12 is power, not particle index
        "pot12 = rsc12_12 * (rsc12_2 - 1.0) / emin12 + 1;"
        "rsc12_12 = rsc12_4 * rsc12_4 * rsc12_4;"
        "rsc12_4 = rsc12_2 * rsc12_2;"
        "rsc12_2 = rsc12_1 * rsc12_1;"
        "rsc12_1 = r12 / REPsigma * rmin12;"
        # calculate distances between particles
        "r12 = distance(p1, p2);"
        "r13 = distance(p1, p3);"
        "r23 = distance(p2, p3);"
    )

    force = mm.CustomManyParticleForce(3, energy)
    force.name = name  # type: ignore

    # set cutoff distance for the force
    force.setCutoffDistance(attractionRadius * sim_object.conlen)

    force.addGlobalParameter("ATTRe", attractionEnergy * sim_object.kT)
    force.addGlobalParameter("REPsigma", attractionRadius * sim_object.conlen)

    # Coefficients for the minimum of x^12*(x*x-1)
    force.addGlobalParameter("emin12", 46656.0 / 823543.0)
    force.addGlobalParameter("rmin12", np.sqrt(6.0 / 7.0))

    # This is a single permutation force, which means that all 3 particles are equivalent
    # The equation has to be symmetric in all 3 particles
    force.setPermutationMode(mm.CustomManyParticleForce.SinglePermutation)
    # try the other permutation for debug
    # force.setPermutationMode(mm.CustomManyParticleForce.UniqueCentralParticle)

    # Need to set up particle filters so 3 types are different
    # Type zero will be "any other particle", and types 1,2,3 are the 3 types in question
    force.setTypeFilter(0, {1})
    force.setTypeFilter(1, {2})
    force.setTypeFilter(2, {3})

    # We need to add all particles to the force, giving other particles a type of 0

    # Make sure index lists are unique
    ind1 = set(type1_particle_idx)
    ind2 = set(type2_particle_idx)
    ind3 = set(type3_particle_idx)

    # verify that lists don't overlap

    if len(ind1 & ind2) > 0 or len(ind1 & ind3) > 0 or len(ind2 & ind3) > 0:
        raise ValueError("Particle indices for the three types must be unique and non-overlapping.")

    # add all particles to the force - even those not in the attraction

    for i in range(sim_object.N):
        particle_type: int = 0  # default type
        if i in ind1:
            particle_type = 1
        elif i in ind2:
            particle_type = 2
        elif i in ind3:
            particle_type = 3

        # no keyword arguments for this one, first one is parameters, second is type
        force.addParticle([], particle_type)  # type: ignore[no-untyped-call]

    return force


def add_fucking_complicated_force(
    sim_object: polychrom.simulation.Simulation, k_linear: float
) -> mm.CustomCompoundBondForce:
    """
    Add a force that drives two dihedrals toward the same target angle, defined by a torque-sensor dihedral.

    The use case for this force is to control bending of SCC1/SCC3 polymers using torque applied to them.
    A "torque sensor" is a "row" in the SCC1/SCC3 that can rotate freely within a certain range
    """

    # Building the torque-sensing structure near the break point
    # break_idx - 3 through break_idx - 2 the dihedral is missing

    # particle indices:
    # ( break_idx - 3, 0): 1     (break_idx, 0): 7
    # ( break_idx - 3, 1): 2     (break_idx, 1): 8
    # ( break_idx - 2, 0): 3     (break_idx + 1, 0): 9
    # ( break_idx - 2, 1): 4     (break_idx + 1 , 1): 10
    # ( break_idx - 1, 0): 5
    # ( break_idx - 1, 1): 6

    # alpha 5/3pi is the "straight line" dihedral for cos(dihedral(p6, p8, p7, p10) - alpha))
    # alpha = 2/3 pi is the "Fully scrunched" angle

    # The torque-sensing part will be simple dihedral between two edges extending from the backbone.
    # Remember that OpenMM expressions are read backwards -  like highway signs, not like text in Star Wars.
    # The final expression is first, and variables it is made of are below.

    expr = """
    // 11) Total energy = hinge-driving + penalty
    E_hinge + E_pen;

    // 10) Hinge-driving part (two dihedrals toward the same target)
    // with a slight linear term to keep the hinge angle small (closed)
    E_hinge   = k*(1 - cos(phi_h1 - theta_tgt))
            + k*(1 - cos(phi_h2 - theta_tgt))
            + k_linear * (phi_h1 + phi_h2);
    // 9) Quadratic wall outside the free torsion window |phi_scaled| <= 1
    E_pen     = k*2 * step(delta) * delta*delta;

    // 8) How far outside the window?  delta = abs(phi_scaled) - 1
    delta     = abs(phi_scaled) - 1;

    // 7) Two hinge dihedrals you want to drive
    phi_h1    = dihedral(p6,p8,p7,p10) * break_sign;
    phi_h2    = dihedral(p5,p8,p7,p9) * break_sign;

    // 6) Target hinge angle in [2*pi/3,5*pi/3] 
    // (probably a constant was added to give it a little more "flexibility")
    theta_tgt = 7*pi/6 + (pi/2 + .15)*phi_clamp_sign;

    // 5) Multiply clamped value by coupling_sign to set direction
    phi_clamp_sign = phi_clamp * coupling_sign;

    // 4) Clamp the normalized torque phi_scaled into [-1,1]
    phi_clamp = max(-1, min(1, phi_scaled));

    // 3) Normalize the torque-sensor dihedral by X
    phi_scaled = phi_ts / X;

    // 2) Torque-sensor dihedral (free window |phi_ts| <= X)
    phi_ts    = dihedral(p2,p1,p3,p4);

    // 1) Constants  - more X means more free rotation
    X         = 1; 
    pi        = 3.141592653589793;
    """
    # remove lines not containing "//"
    expr = "\n".join(line for line in expr.splitlines() if not "//" in line).strip()

    fuckingComplicatedForce = mm.CustomCompoundBondForce(10, expr)
    fuckingComplicatedForce.name = "fuckingComplicatedForce"  # type: ignore - this is polychrom's old convention
    fuckingComplicatedForce.addGlobalParameter("k", 200 * sim_object.kT)
    fuckingComplicatedForce.addGlobalParameter("k_linear", k_linear * sim_object.kT)
    fuckingComplicatedForce.addPerBondParameter("coupling_sign")
    fuckingComplicatedForce.addPerBondParameter("break_sign")
    return fuckingComplicatedForce
