"""Test xcube tracing behavior with many subvariables."""
import numpy

from catii import xcube
from catii.xfuncs import xfunc_count, xfunc_stddev


class TestXCubeTracing:
    def test_tracing_start_set_with_1d_dimensions(self):
        """Test that tracing start is set when dimensions are 1D."""
        arr1 = numpy.array([1, 0, 1, 0, 0, 0, 0, 1], dtype=numpy.int8)
        arr2 = numpy.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=numpy.int8)
        
        cube = xcube([arr1, arr2])
        funcs = [xfunc_count()]
        cube.calculate(funcs)
        
        # Check that tracing was recorded
        for f in funcs:
            info = cube._tracing[f]
            assert info["start"] is not None, "start should be set"
            assert info["count"] > 0, "count should be > 0"
            assert info["elapsed"] > 0, "elapsed should be > 0"

    def test_tracing_start_set_with_2d_dimension(self):
        """Test that tracing start is set when one dimension is 2D (has subvariables)."""
        # 1D dimension with 3 categories
        arr1 = numpy.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=numpy.int8)
        # 2D dimension with shape (rows, 5 subvars)
        arr2 = numpy.array([
            [0, 1, 0, 1, 2],
            [1, 0, 1, 2, 0],
            [2, 1, 2, 0, 1],
            [0, 2, 0, 1, 2],
            [1, 0, 1, 2, 0],
            [2, 1, 2, 0, 1],
            [0, 2, 0, 1, 2],
            [1, 0, 1, 2, 0],
        ], dtype=numpy.int8)
        
        cube = xcube([arr1, arr2], interacting_shape=(3, 3))
        funcs = [xfunc_count()]
        cube.calculate(funcs)
        
        # Check product has expected number of items (5 subvariables)
        product_list = list(cube.product)
        assert len(product_list) == 5, f"Expected 5 products, got {len(product_list)}"
        
        # Check that tracing was recorded
        for f in funcs:
            info = cube._tracing[f]
            assert info["start"] is not None, "start should be set"
            assert info["count"] == 5, f"count should be 5, got {info['count']}"
            assert info["elapsed"] > 0, "elapsed should be > 0"

    def test_tracing_start_with_many_subvariables(self):
        """Test that tracing start is set with 269 subvariables (production case)."""
        num_rows = 1000
        num_cats = 3
        num_subvars = 269
        
        # Create dimensions matching production case:
        # - arr1: 1D category dimension, shape (rows,)
        # - But the actual dimension passed to xcube has shape (rows, num_subvars)
        #   because it represents an array variable
        
        # Single 2D dimension with many subvariables
        arr = numpy.random.randint(0, num_cats, size=(num_rows, num_subvars), dtype=numpy.int8)
        
        cube = xcube([arr], interacting_shape=(num_cats,))
        
        # Verify product generates correct number of items
        product_list = list(cube.product)
        assert len(product_list) == num_subvars, f"Expected {num_subvars} products, got {len(product_list)}"
        
        funcs = [xfunc_count()]
        cube.calculate(funcs)
        
        # Check that tracing was recorded
        for f in funcs:
            info = cube._tracing[f]
            assert info["start"] is not None, "start should be set with many subvariables"
            assert info["count"] == num_subvars, f"count should be {num_subvars}, got {info['count']}"
            assert info["elapsed"] > 0, "elapsed should be > 0"

    def test_tracing_with_stddev_many_subvariables(self):
        """Test stddev tracing with many subvariables."""
        num_rows = 1000
        num_cats = 3
        num_subvars = 50  # Use fewer for faster test
        
        values = numpy.random.randn(num_rows, num_subvars).astype(numpy.float64)
        coords = numpy.random.randint(0, num_cats, size=(num_rows, num_subvars), dtype=numpy.int8)
        
        cube = xcube([coords], interacting_shape=(num_cats,))
        
        funcs = [xfunc_stddev(values)]
        cube.calculate(funcs)
        
        # Check that tracing was recorded
        for f in funcs:
            info = cube._tracing[f]
            assert info["start"] is not None, "start should be set for stddev"
            assert info["count"] == num_subvars, f"count should be {num_subvars}, got {info['count']}"
            assert info["elapsed"] > 0, "elapsed should be > 0"

    def test_product_empty_with_scalar_dimension(self):
        """Test that product is not empty even with special dimension shapes."""
        # This tests edge cases where product might be unexpectedly empty
        
        # Scalar dimension (no shape beyond rows)
        arr = numpy.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=numpy.int8)
        
        cube = xcube([arr])
        product_list = list(cube.product)
        
        # Even with a simple 1D array, product should have at least 1 item (None,)
        assert len(product_list) >= 1, f"Product should not be empty, got {len(product_list)}"

    def test_tracing_parallel_many_subvariables(self):
        """Test tracing with parallel execution and many subvariables (like production)."""
        from multiprocessing.pool import ThreadPool
        
        num_rows = 10000  # Smaller than production for speed, but enough to test
        num_cats = 3
        num_subvars = 269  # Match production case
        
        # Create dimension matching production case
        coords = numpy.random.randint(0, num_cats, size=(num_rows, num_subvars), dtype=numpy.int8)
        values = numpy.random.randn(num_rows, num_subvars).astype(numpy.float64)
        
        cube = xcube([coords], interacting_shape=(num_cats,))
        cube.pool_class = ThreadPool
        cube.poolsize = 4
        
        funcs = [xfunc_stddev(values)]
        cube.calculate(funcs)
        
        # Check that tracing was recorded even with parallel execution
        for f in funcs:
            info = cube._tracing[f]
            assert info["start"] is not None, f"start should be set with parallel execution, got {info}"
            assert info["count"] == num_subvars, f"count should be {num_subvars}, got {info['count']}"
            assert info["elapsed"] > 0, f"elapsed should be > 0, got {info['elapsed']}"

    def test_tracing_serial_execution_269_subvars(self):
        """Test that serial execution with 269 subvars records tracing.
        
        This reproduces the production case where:
        - 561182 rows
        - 269 subvariables
        - scaffold_size = 269
        - rows * scaffold_size = 151M < BIG_REGIONS (1B)
        - So parallel=False, runs serially
        
        In production, info["start"] was None, meaning tracing wasn't recorded.
        """
        from catii.xcubes import BIG_REGIONS
        
        num_rows = 10000  # Smaller for test speed, but enough to verify
        num_cats = 3
        num_subvars = 269
        
        coords = numpy.random.randint(0, num_cats, size=(num_rows, num_subvars), dtype=numpy.int8)
        values = numpy.random.randn(num_rows, num_subvars).astype(numpy.float64)
        
        cube = xcube([coords], interacting_shape=(num_cats,))
        
        # Verify this is a serial execution case (scaffold_size too small for parallel)
        assert cube.scaffold_size == num_subvars
        expected_parallel = (
            cube.scaffold_size > 2
            and (coords.shape[0] * cube.scaffold_size) >= BIG_REGIONS
        )
        # For test: 10000 * 269 = 2.69M << 1B, so serial
        assert not expected_parallel, "Test expects serial execution"
        assert not cube.parallel, f"Expected serial execution, got parallel={cube.parallel}"
        
        funcs = [xfunc_stddev(values), xfunc_count()]
        cube.calculate(funcs)
        
        # Verify all funcs have tracing recorded
        for i, f in enumerate(funcs):
            info = cube._tracing[f]
            assert info["start"] is not None, f"start should be set for func {i}: {info}"
            assert info["count"] == num_subvars, f"count should be {num_subvars}, got {info['count']}"
            assert info["elapsed"] > 0, f"elapsed should be > 0, got {info['elapsed']}"

    def test_tracing_mimics_zz9d_usage(self):
        """Test xcube usage pattern that matches zz9d RectangularAlgorithm.
        
        In zz9d:
        1. cube = xcube([dim.entries for dim in self.interacting_dims], interacting_shape=...)
        2. cube.pool_class = runtime.job.pool  (a method, not a class)
        3. cube.check_interrupt = self.check_interrupt
        4. funcs = [self.measures[k].xfunc() for k in measure_names]
        5. final_results = cube.calculate(funcs)
        """
        num_rows = 10000
        num_cats = 3
        num_subvars = 50  # Faster test
        
        # Simulate dim.entries returning 2D array for indexed data
        entries = numpy.random.randint(0, num_cats, size=(num_rows, num_subvars), dtype=numpy.int8)
        
        # This is what zz9d does: wraps in a list, passes to xcube
        cube = xcube([entries], interacting_shape=(num_cats,))
        
        # Set pool_class like zz9d does (to a method, but won't be used for serial)
        # In production this is runtime.job.pool
        def mock_pool(poolsize):
            from multiprocessing.pool import ThreadPool
            return ThreadPool(poolsize)
        cube.pool_class = mock_pool
        
        # Set check_interrupt like zz9d does
        def check_interrupt():
            pass  # No-op for test
        cube.check_interrupt = check_interrupt
        
        # Create funcs like zz9d does (new xfunc instances)
        values = numpy.random.randn(num_rows, num_subvars).astype(numpy.float64)
        funcs = [xfunc_stddev(values), xfunc_count()]
        
        # Calculate like zz9d does
        final_results = cube.calculate(funcs)
        
        # Verify tracing works
        for i, f in enumerate(funcs):
            info = cube._tracing[f]
            assert info["start"] is not None, f"start should be set for func {i}: {info}"
            assert info["count"] == num_subvars, f"count should be {num_subvars}, got {info['count']}"
            assert info["elapsed"] > 0, f"elapsed should be > 0, got {info['elapsed']}"
        
        # Verify results are returned
        assert len(final_results) == 2
        assert final_results[0] is not None  # stddev result
        assert final_results[1] is not None  # count result

    def test_tracing_parallel_vs_serial_consistency(self):
        """Test that parallel and serial execution produce same tracing count."""
        from multiprocessing.pool import ThreadPool
        
        num_rows = 1000
        num_cats = 3
        num_subvars = 50
        
        numpy.random.seed(42)  # Reproducibility
        coords = numpy.random.randint(0, num_cats, size=(num_rows, num_subvars), dtype=numpy.int8)
        
        # Serial execution
        cube_serial = xcube([coords.copy()], interacting_shape=(num_cats,))
        funcs_serial = [xfunc_count()]
        cube_serial.calculate(funcs_serial)
        serial_info = cube_serial._tracing[funcs_serial[0]]
        
        # Parallel execution  
        cube_parallel = xcube([coords.copy()], interacting_shape=(num_cats,))
        cube_parallel.pool_class = ThreadPool
        cube_parallel.poolsize = 4
        funcs_parallel = [xfunc_count()]
        cube_parallel.calculate(funcs_parallel)
        parallel_info = cube_parallel._tracing[funcs_parallel[0]]
        
        # Both should have same count
        assert serial_info["count"] == parallel_info["count"], \
            f"Serial count {serial_info['count']} != parallel count {parallel_info['count']}"
        assert serial_info["count"] == num_subvars
        assert serial_info["start"] is not None
        assert parallel_info["start"] is not None
