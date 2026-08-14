# Writing MPI code with climtools

This is the guide to writing Python that uses the MPI layer. For building the
native extension, see [BUILD.md](BUILD.md).

The model is deliberately narrow. There is no communicator algebra, no
non-blocking sends, no custom datatypes. There is one world, every rank runs
the same script, and the two things you actually need for data analysis are
made easy: **split the work along one axis**, and **write one shared file**.

Nothing here initializes MPI until you call something that needs it, so
`import climtools` costs nothing in a serial session.

---

## The shape of an MPI script

Every rank runs the whole file, top to bottom. There is no controller process
handing out work. Ranks differ only in what `mpi.rank()` returns, and
everything else follows from that.

The consequence that catches people out: **every rank must reach every
collective call, in the same order**. A collective is any operation that needs
all ranks to participate, which here means anything that reduces, gathers,
broadcasts, waits, or writes. If one rank takes an `if` branch that skips a
collective, the others wait for it forever and the job hangs until the
scheduler kills it.

This layer removes most of the ways to get that wrong. Decorate a function and
the wrapper handles the collective for you, from every rank, in the right
order.

---

## Step 1. Import

```python
from climtools import MPI          # the decorator
from climtools import lib_mpi as mpi   # rank(), size(), total(), barrier(), ...
```

`MPI` declares *where* a function runs. `mpi.*` answers *who am I* and does the
reductions.

---

## Step 2. Decide where each function runs

This is the whole design. Every function falls into one of three cases.

### Runs everywhere: `@MPI(all_ranks=True)`

For anything that computes. Each rank works on its own data, and if any rank
raises, every rank is told and the job fails as a unit instead of half
finishing.

```python
@MPI(all_ranks=True)
def compute_statistics(local):
    return local.mean("time")
```

### Runs on the root only: `@MPI`

For anything that touches the filesystem: making directories, deleting stale
output, copying results, logging a summary. Doing these on every rank means
eight processes racing to create the same directory.

```python
@MPI
def prepare_output(path):
    path.mkdir(parents=True, exist_ok=True)
```

Every rank still *calls* `prepare_output(...)`. The wrapper is collective: the
non-root ranks enter it and wait, so they cannot run ahead and open a file in
a directory that does not exist yet. Non-root ranks get `None` back. If you
need the return value everywhere, use `@MPI(broadcast=True)`.

```python
@MPI(broadcast=True)
def read_config(path):
    return json.loads(path.read_text())   # every rank gets the dict
```

Reading a shared file from one rank and broadcasting beats a thousand ranks
opening the same file at once, which is a well-known way to bring a parallel
filesystem to its knees.

### Runs locally: no decorator

Pure functions that neither compute across ranks nor touch shared state need
nothing. Most of your code should be in this category.

```python
def land_mask(ds):
    return ds["slmsk"].squeeze(drop=True) == 1
```

**Rule of thumb:** decorate for side effects and for failure propagation.
If a function is pure and local, leave it alone.

---

## Step 3. Split the work

Use `mpi.partition(n)` to get this rank's half-open block of `n` items.

```python
start, stop = mpi.partition(dataset.sizes["event"])
local = dataset.isel(event=slice(start, stop))
```

Or, on an xarray object, in one call:

```python
local = dataset.xgeo.mpi.partition("event")
```

Blocks are **contiguous**, and the remainder falls on the leading ranks, so
lengths differ by at most one. Contiguity is not cosmetic: the parallel writer
recovers each rank's offset in the output file from the local lengths, so a
strided or round-robin split would scatter each rank's records across the
whole file.

The other helpers:

| Call | Does |
| ---- | ---- |
| `mpi.partition(n)` | this rank's `(start, stop)` bounds |
| `mpi.split(seq)` | this rank's slice of a list or array |
| `mpi.scatter(items)` | root hands one item to each rank |

`partition` and `split` communicate nothing. Every rank computes the same
split independently and takes its own piece, which is the cheap pattern when
the data is on a shared filesystem and every rank can open it lazily.

---

## Step 4. Combine results

A reduction turns one value per rank into one value on every rank.

```python
total   = mpi.total(local_sum)        # sum across ranks
largest = mpi.maximum(local_max)
mean    = mpi.mean(local_value)       # equal weight per rank
```

The same operations hang off xarray objects:

```python
combined = local_partial.xgeo.mpi.sum()
```

Available: `sum`/`total`, `prod`, `min`/`minimum`, `max`/`maximum`, `any`,
`all`, `mean`, plus `mpi.reduce(value, op)` if you want to pass the operator by
name.

### The one thing to get right: means are not additive

You cannot average per rank and then average the averages. A rank holding
three events would count as much as a rank holding three thousand.

A weighted mean is a **ratio of two sums**, so reduce the numerator and the
denominator separately and divide at the end:

```python
local_numerator   = (weights * values).sum("event")
local_denominator = weights.sum("event")

numerator   = mpi.total(local_numerator)
denominator = mpi.total(local_denominator)

composite = numerator / denominator          # correct at any rank count
```

`mpi.mean()` is the equal-weight-per-rank mean, which is the right tool only
when every rank contributes exactly one comparable value.

The same reasoning applies to any statistic that is not a sum. Counts, sums
and sums of squares reduce cleanly; means, variances and correlations must be
rebuilt from those. Minima and maxima reduce directly.

### Reproducibility

Reductions run in rank order, so every rank gets a **bit-identical** result.
The parallel writer depends on this: it checks that arrays it treats as
replicated agree across ranks, and refuses the write if they differ by one
bit.

Results are *not* bit-identical across different rank counts. Partitioning
changes the order in which partial sums are associated, and floating-point
addition is not associative. The differences are at the level of a few
thousand times machine epsilon, far below anything physically meaningful, but
do not expect two jobs at different widths to produce byte-identical files.

---

## Step 5. Write one file, collectively

Every rank contributes its slab and the file is written once. Nothing is
gathered to rank zero, and there are no per-rank files to merge afterwards.

```python
local.xgeo.mpi.to_netcdf("events.nc", partition_dim="event")
```

or through the function interface:

```python
xgeo.to_netcdf(
    data=local,
    file="events.nc",
    partition_dim="event",
    parallel=True,
    allow_serial=True,
)
```

What every rank must agree on: variable names, dtypes, dimension names, and
attributes. What may differ: the length of `partition_dim`, including zero.

A rank that drew an empty block still has to call the write. It contributes a
zero-length slab and stays in the collective. This happens whenever there are
fewer records than ranks, and it is the single most common way a
nearly-correct script hangs.

Variables *without* `partition_dim` are treated as replicated. The writer
fingerprints them and fails loudly if the ranks disagree, rather than silently
writing rank zero's copy.

`allow_serial=True` lets the same call work with one rank, so the script runs
unlaunched.

### Compression

Deflate during a collective write needs a NetCDF-C and HDF5 built with
parallel filters, which many stacks lack. The default is therefore off. If you
ask for it on a stack that cannot do it, you get a warning and an uncompressed
file; pass `strict_compression=True` to make that an error instead. Check with:

```python
import climtools.lib_mpi as mpi
print(mpi.info())
```

---

## Step 6. Run it

```bash
python script.py                              # one rank, no launcher
mpirun -n 8 python script.py
srun --ntasks=8 --mpi=pmix python script.py   # Slurm
```

Use the same MPI and NetCDF stack that `install.sh` verified. Mixing a second
MPI implementation, or a serial HDF5, into the runtime environment is the
usual cause of an inexplicable crash. `build/build.yml` records exactly what
was verified.

Match `--ntasks` to the work, not to the machine. Splitting 1500 events over
2000 ranks gives 500 ranks a zero-length block and a great deal of collective
overhead for nothing.

---

## A complete example

```python
from pathlib import Path

import xarray as xr

from climtools import MPI
from climtools import lib_mpi as mpi


@MPI
def prepare(output: Path) -> None:
    """Root only. Other ranks wait here until the directory exists."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)


@MPI(all_ranks=True)
def weighted_composite(local: xr.Dataset) -> xr.Dataset:
    """Cosine-latitude weighted mean, reduced as a ratio of two sums."""
    weights = np.cos(np.deg2rad(local["lat"]))
    numerator = (local * weights).sum("event")
    denominator = (weights.broadcast_like(local) .where(local.notnull())).sum("event")
    return mpi.total(numerator) / mpi.total(denominator)


@MPI(all_ranks=True)
def main() -> None:
    output = Path("composites/events.nc")
    prepare(output)
    mpi.barrier()

    with xr.open_dataset("history.nc", chunks={}) as source:
        local = source.xgeo.mpi.partition("event")

        if mpi.is_root():
            print(f"{source.sizes['event']} events over {mpi.size()} ranks")

        composite = weighted_composite(local)
        store = xr.merge([local, composite.rename({v: f"{v}_composite"
                                                   for v in composite.data_vars})])

        store.xgeo.mpi.to_netcdf(output, partition_dim="event")

    mpi.barrier()


if __name__ == "__main__":
    main()
```

See [`../examples/time_composites.py`](../examples/time_composites.py) for the
same pattern on a real workload.

---

## Reference

### Decorator

| Form | Runs on | Returns |
| ---- | ------- | ------- |
| `@MPI` | root only | value on root, `None` elsewhere |
| `@MPI(broadcast=True)` | root only | value on every rank |
| `@MPI(all_ranks=True)` | every rank | that rank's value |
| `@MPI(root=1)` | rank 1 only | value on rank 1 |
| `@MPI(require_ranks=4)` | as configured | fails unless the world has at least 4 ranks |

### Identity

`mpi.rank()`, `mpi.size()`, `mpi.is_root()`, `mpi.comm()`

### Communication

`mpi.barrier()`, `mpi.bcast(obj, root=0)`, `mpi.gather(obj, root=0)`,
`mpi.allgather(obj)`, `mpi.scatter(items, root=0)`, `mpi.consensus(ok)`

### Reduction

`mpi.total()`, `mpi.prod()`, `mpi.minimum()`, `mpi.maximum()`, `mpi.mean()`,
`mpi.reduce(value, op)`

### Decomposition

`mpi.partition(n)`, `mpi.split(sequence)`

### On xarray objects

Reached as `ds.xgeo.mpi.<...>` or `ds.mpi.<...>`:

`sum`, `prod`, `min`, `max`, `any`, `all`, `mean`, `reduce`, `bcast`,
`gather`, `allgather`, `concat`, `scatter`, `partition`, `barrier`,
`to_netcdf`

### Diagnostics

```python
mpi.info()                   # library path, NetCDF version, ABI, filters, world
mpi.available()              # is the compiled extension usable
mpi.has_parallel_filters()   # can a collective write compress
```

---

## When something hangs

Almost always a rank that skipped a collective. In order of likelihood:

1. A `return` or `continue` inside `if mpi.is_root():` that jumps over a
   later collective. Take the decision collectively instead, from a value all
   ranks share, or use `mpi.consensus()`.
2. A rank with an empty block skipping the write. It must still call it.
3. A `try`/`except` that swallows an error on one rank so it carries on into a
   collective the others have already left. Use `@MPI(all_ranks=True)`, which
   propagates the failure to everyone.
4. Different iteration counts per rank, from looping over something that is
   not identical everywhere.

If a rank raises, the writer aborts the whole job rather than leaving the
others in a collective. An abort is not a bug in your script by itself; read
the traceback above it, which names the rank that failed first.
