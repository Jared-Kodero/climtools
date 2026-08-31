"""Smoke-test mpi4py Cartcomm.Shift() face-neighbor semantics directly.

Confirms the raw MPI behavior climtools.xarray.cartesian.CartesianTopology
is built on, independent of any climtools code, before trusting it.

Run: mpirun -n 4 python3 test_cartesian_topology_smoke.py
"""

from mpi4py import MPI

comm = MPI.COMM_WORLD
cart = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
coords = cart.Get_coords(cart.rank)
lo0, hi0 = cart.Shift(0, 1)
lo1, hi1 = cart.Shift(1, 1)
print(
    f"rank={comm.rank} coords={coords} axis0(lo,hi)=({lo0},{hi0}) "
    f"axis1(lo,hi)=({lo1},{hi1}) PROC_NULL={MPI.PROC_NULL}"
)
