import sys
import atexit


MPI = None

def _import_mpi(use_dill=False):
    global MPI
    try:
        from mpi4py import MPI as _MPI
        if use_dill:
            import dill
            _MPI.pickle.__init__(dill.dumps, dill.loads, dill.HIGHEST_PROTOCOL)
        MPI = _MPI
    except:
        raise ImportError("Please install mpi4py")

    return MPI


class MPIPool:
    """A processing pool that distributes tasks using MPI.
    With this pool class, the master process distributes tasks to worker
    processes using an MPI communicator.
    This implementation is inspired by @juliohm in `this module
    <https://github.com/juliohm/HUM/blob/master/pyhum/utils.py#L24>`_
    and was adapted from schwimmbad.
    Parameters
    ----------
    comm : :class:`mpi4py.MPI.Comm`, optional
        An MPI communicator to distribute tasks with. If ``None``, this uses
        ``MPI.COMM_WORLD`` by default.
    """

    def __init__(self, comm=None):

        self.comm = MPI.COMM_WORLD if comm is None else comm

        self.master = 0
        self.rank = self.comm.Get_rank()

        atexit.register(lambda: MPIPool.close(self))

        if not self.is_master():
            # workers branch here and wait for work
            self.wait()
            sys.exit(0)

        self.workers = set(range(self.comm.size))
        self.workers.discard(self.master)
        self.size = self.comm.Get_size() - 1

        if self.size == 0:
            raise ValueError("Tried to create an MPI pool, but there "
                             "was only one MPI process available. "
                             "Need at least two.")


    def wait(self):
        """Tell the workers to wait and listen for the master process. This is
        called automatically when using :meth:`MPIPool.map` and doesn't need to
        be called by the user.
        """
        if self.is_master():
            return

        status = MPI.Status()
        while True:
            task = self.comm.recv(source=self.master, tag=MPI.ANY_TAG, status=status)

            if task is None:
                # Worker told to quit work
                break

            func, arg = task
            result = func(arg)
            # Worker is sending answer with tag
            self.comm.ssend(result, self.master, status.tag)


    def map(self, worker, tasks):
        """Evaluate a function or callable on each task in parallel using MPI.
        The callable, ``worker``, is called on each element of the ``tasks``
        iterable. The results are returned in the expected order.
        Parameters
        ----------
        worker : callable
            A function or callable object that is executed on each element of
            the specified ``tasks`` iterable. This object must be picklable
            (i.e. it can't be a function scoped within a function or a
            ``lambda`` function). This should accept a single positional
            argument and return a single object.
        tasks : iterable
            A list or iterable of tasks. Each task can be itself an iterable
            (e.g., tuple) of values or data to pass in to the worker function.

        Returns
        -------
        results : list
            A list of results from the output of each ``worker()`` call.
        """

        # If not the master just wait for instructions.
        if not self.is_master():
            self.wait()
            return


        workerset = self.workers.copy()
        tasklist = [(tid, (worker, arg)) for tid, arg in enumerate(tasks)]
        resultlist = [None] * len(tasklist)
        pending = len(tasklist)

        while pending:
            if workerset and tasklist:
                worker = workerset.pop()
                taskid, task = tasklist.pop()
                # "Sent task %s to worker %s with tag %s"
                self.comm.send(task, dest=worker, tag=taskid)

            if tasklist:
                flag = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                if not flag:
                    continue
            else:
                self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

            status = MPI.Status()
            result = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                    status=status)
            worker = status.source
            taskid = status.tag

            # "Master received from worker %s with tag %s"

            workerset.add(worker)
            resultlist[taskid] = result
            pending -= 1

        return resultlist


    def close(self):
        """ Tell all the workers to quit."""
        if self.is_worker():
            return

        for worker in self.workers:
            self.comm.send(None, worker, 0)


    def is_master(self):
        return self.rank == 0


    def is_worker(self):
        return self.rank != 0


    def __enter__(self):
        return self


    def __exit__(self, *args):
        self.close()



def split_ranks(N_ranks, N_chunks):
    """
    Divide the ranks into N chunks
    """
    seq = range(N_ranks)
    avg = int(N_ranks // N_chunks)
    remainder = N_ranks % N_chunks

    start = 0
    end = avg
    for i in range(N_chunks):
        if remainder:
            end += 1
            remainder -= 1
        yield i, seq[start:end]
        start = end
        end += avg



class ChainManager:
    """
    Class to serve as context manager to handle to MPI-related issues,
    specifically, the managing of ``MPIPool`` and splitting of communicators.
    This class can be used to run ``nchains`` in parallel with each chain
    having its own ``MPIPool`` of parallel walkers.
    
    Parameters
    ----------
    nchains : int
        the number of independent chains to run concurrently
    comm : MPI.Communicator
        the global communicator to split
    """

    def __init__(self, nchains=1, comm=None):
        global MPI
        MPI = _import_mpi(use_dill=False)

        self.comm  = MPI.COMM_WORLD if comm is None else comm
        self.nchains = nchains

        # initialize comm for parallel chains
        self.chains_group = None
        self.chains_comm  = None

        # intiialize comm for pool of workers for each parallel chain
        self.pool_comm = None
        self.pool      = None


    def __enter__(self):
        """
        Setup the MPIPool, such that only the ``pool`` master returns,
        while the other processes wait for tasks
        """
        # split ranks if we need to
        if self.comm.size > 1:

            ranges = []
            for i, ranks in split_ranks(self.comm.size, self.nchains):
                ranges.append(ranks[0])
                if self.comm.rank in ranks:
                    color = i

            # split the global comm into pools of workers
            self.pool_comm = self.comm.Split(color, 0)

            # make the comm to communicate b/w parallel runs
            if self.nchains >= 1:
                self.chains_group = self.comm.group.Incl(ranges)
                self.chains_comm = self.comm.Create(self.chains_group)

        # initialize the MPI pool, if the comm has more than 1 process
        if self.pool_comm is not None and self.pool_comm.size > 1:
            self.pool = MPIPool(comm=self.pool_comm)

        # explicitly force non-master ranks in pool to wait
        if self.pool is not None and not self.pool.is_master():
            self.pool.wait()
            sys.exit(0)

        self.rank = 0
        if self.chains_comm is not None:
            self.rank = self.chains_comm.rank

        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Exit gracefully by closing and freeing the MPI-related variables
        """
        # wait for all the processes, if we more than one
        if self.chains_comm is not None and self.chains_comm.size > 1:
            self.chains_comm.Barrier()

        # close and free the MPI stuff
        if self.chains_group is not None:
            self.chains_group.Free()
        if self.chains_comm is not None:
            self.chains_comm.Free()
        if self.pool is not None:
            self.pool.close()


    @property
    def get_rank(self):
        '''
        Get ``rank`` of current ``chain``. The minimum ``rank`` is ``0`` and the maximum is ``nchains-1``.
        '''
        return self.rank


    @property
    def get_pool(self):
        '''
        Get parallel ``pool`` of workers that correspond to a specific chain. This should be used to
        parallelize the walkers of each ``chain`` (not the chains themselves). This includes the ``map``
        method that ``zeus`` requires.
        '''
        return self.pool

    
    def gather(self, x, root):
        '''
        Gather method to gather ``x`` in ``rank = root`` chain.

        Parameters
        ----------
        x : Python object
            The python object to be gathered.
        root : int
            The rank  of the chain that x is gathered.

        Returns
        -------
        x : Python object
            The input object x gathered in ``rank = root``.
        '''
        return self.chains_comm.gather(x, root=root)

    
    def scatter(self, x, root):
        '''
        Scatter method to scatter ``x`` from ``rank = root`` chain to the rest. 

        Parameters
        ----------
        x : Python object
            The python object to be scattered.
        root : int
            The rank of the origin chain from which the x is scattered.

        Returns
        -------
        x : Pythonn object
            Part of the input object x that was scattered along the ranks.
        '''
        return self.chains_comm.scatter(x, root=root)


    def allgather(self, x):
        '''
        Allgather method to gather ``x`` in all chains. This is equivalent to first ``scatter`` and then ``bcast``.

        Parameters
        ----------
        x : Python object
            The python object to be gathered.

        Returns
        -------
        x : Python object
            The python object, gathered in all ranks.
        '''
        return self.chains_comm.allgather(x)


    def bcast(self, x, root):
        '''
        Broadcast method to send ``x`` from ``rank = root`` to all chains.

        Parameters
        ----------
        x : Python object
            The python object to be send.
        root : int
            The rank of the origin chain from which the object x is sent.
        
        Returns
        -------
        x : Python object
            The input object x in all ranks.
        '''
        return self.chains_comm.bcast(x, root=root)
