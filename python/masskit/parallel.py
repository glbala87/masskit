"""
Parallel and batch processing for LC-MS data.

Provides multi-file batch processing, parallel spectrum processing,
and convenience functions for common parallel workflows.
"""

from typing import Optional, List, Dict, Callable, Any, Tuple
from dataclasses import dataclass
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

from .spectrum import Spectrum
from .peak import PeakList
from .feature import FeatureMap


@dataclass
class BatchResult:
    """Result from processing a single file."""
    filepath: str
    result: Any = None
    error: Optional[str] = None
    success: bool = True


class BatchProcessor:
    """
    Parallel batch processor for LC-MS files.

    Processes multiple files concurrently using multiprocessing or threading.

    Example:
        >>> processor = BatchProcessor(n_workers=4)
        >>> results = processor.process_files(
        ...     file_paths, workflow=my_workflow
        ... )
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        backend: str = "multiprocessing",
    ):
        """
        Args:
            n_workers: Number of parallel workers (None = all CPUs)
            backend: 'multiprocessing' or 'threading'
        """
        self.n_workers = n_workers or multiprocessing.cpu_count()
        self.backend = backend

    def _get_executor(self):
        if self.backend == "threading":
            return ThreadPoolExecutor(max_workers=self.n_workers)
        return ProcessPoolExecutor(max_workers=self.n_workers)

    def process_files(
        self,
        file_paths: List[str],
        workflow: Callable,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        **kwargs,
    ) -> List[BatchResult]:
        """
        Process multiple files in parallel.

        Args:
            file_paths: List of file paths to process
            workflow: Function that takes (filepath, **kwargs) and returns a result
            on_progress: Optional callback(completed, total, current_file)
            **kwargs: Additional arguments passed to workflow

        Returns:
            List of BatchResult objects
        """
        results = []
        total = len(file_paths)

        with self._get_executor() as executor:
            futures = {
                executor.submit(_process_file_wrapper, fp, workflow, kwargs): fp
                for fp in file_paths
            }

            completed = 0
            for future in as_completed(futures):
                filepath = futures[future]
                completed += 1

                try:
                    result = future.result()
                    results.append(BatchResult(
                        filepath=filepath,
                        result=result,
                        success=True,
                    ))
                except Exception as e:
                    results.append(BatchResult(
                        filepath=filepath,
                        error=str(e),
                        success=False,
                    ))

                if on_progress:
                    on_progress(completed, total, filepath)

        return results

    def map(
        self,
        func: Callable,
        items: List[Any],
        **kwargs,
    ) -> List[Any]:
        """
        Apply function to items in parallel.

        Args:
            func: Function to apply
            items: List of items
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        with self._get_executor() as executor:
            if kwargs:
                futures = [
                    executor.submit(func, item, **kwargs) for item in items
                ]
            else:
                futures = [executor.submit(func, item) for item in items]
            return [f.result() for f in futures]

    def process_spectra(
        self,
        spectra: List[Spectrum],
        func: Callable,
        **kwargs,
    ) -> List[Any]:
        """
        Process spectra in parallel.

        Args:
            spectra: List of Spectrum objects
            func: Function to apply to each spectrum
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        return self.map(func, spectra, **kwargs)


class ParallelSpectrumProcessor:
    """
    Process spectra within a single experiment in parallel.

    Example:
        >>> psp = ParallelSpectrumProcessor(n_workers=4)
        >>> peaks = psp.parallel_peak_pick(experiment, min_snr=5)
    """

    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or multiprocessing.cpu_count()

    def parallel_smooth(
        self,
        spectra: List[Spectrum],
        method: str = "gaussian",
        window_size: int = 5,
        **kwargs,
    ) -> List[Spectrum]:
        """Smooth multiple spectra in parallel."""
        from .algorithms import smooth_spectrum

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(smooth_spectrum, spec, method, window_size, **kwargs)
                for spec in spectra
            ]
            return [f.result() for f in futures]

    def parallel_baseline_correct(
        self,
        spectra: List[Spectrum],
        method: str = "snip",
        **kwargs,
    ) -> List[Spectrum]:
        """Correct baseline for multiple spectra in parallel."""
        from .algorithms import correct_baseline

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(correct_baseline, spec, method, **kwargs)
                for spec in spectra
            ]
            return [f.result() for f in futures]

    def parallel_peak_pick(
        self,
        spectra: List[Spectrum],
        min_snr: float = 3.0,
        **kwargs,
    ) -> List[PeakList]:
        """Pick peaks from multiple spectra in parallel."""
        from .algorithms import pick_peaks

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(pick_peaks, spec, min_snr=min_snr, **kwargs)
                for spec in spectra
            ]
            return [f.result() for f in futures]


def _process_file_wrapper(filepath: str, workflow: Callable, kwargs: dict) -> Any:
    """Wrapper for processing a single file (used by ProcessPoolExecutor)."""
    return workflow(filepath, **kwargs)


def batch_peak_picking(
    file_paths: List[str],
    n_workers: Optional[int] = None,
    **peak_params,
) -> Dict[str, PeakList]:
    """
    Convenience function: pick peaks from multiple files in parallel.

    Args:
        file_paths: List of file paths (mzML or mzXML)
        n_workers: Number of workers
        **peak_params: Parameters passed to pick_peaks

    Returns:
        Dict mapping filepath -> PeakList
    """
    from .io import load_mzml, load_mzxml
    from .algorithms import pick_peaks

    def _process(filepath, **params):
        if filepath.lower().endswith(".mzxml"):
            exp = load_mzxml(filepath)
        else:
            exp = load_mzml(filepath)

        all_peaks = PeakList()
        for i in range(exp.num_spectra):
            spec = exp.spectrum(i)
            peaks = pick_peaks(spec, **params)
            all_peaks.extend(list(peaks))
        return all_peaks

    processor = BatchProcessor(n_workers=n_workers)
    results = processor.process_files(file_paths, _process, **peak_params)

    return {
        r.filepath: r.result
        for r in results
        if r.success
    }


def batch_feature_detection(
    file_paths: List[str],
    n_workers: Optional[int] = None,
    **feature_params,
) -> Dict[str, FeatureMap]:
    """
    Convenience function: detect features from multiple files in parallel.

    Args:
        file_paths: List of file paths
        n_workers: Number of workers
        **feature_params: Parameters passed to FeatureDetectionWorkflow

    Returns:
        Dict mapping filepath -> FeatureMap
    """
    from .io import load_mzml, load_mzxml
    from .workflows import FeatureDetectionWorkflow

    def _process(filepath, **params):
        if filepath.lower().endswith(".mzxml"):
            exp = load_mzxml(filepath)
        else:
            exp = load_mzml(filepath)

        workflow = FeatureDetectionWorkflow(**params)
        return workflow.process(exp)

    processor = BatchProcessor(n_workers=n_workers)
    results = processor.process_files(file_paths, _process, **feature_params)

    return {
        r.filepath: r.result
        for r in results
        if r.success
    }


def batch_process(
    file_paths: List[str],
    pipeline: List[Tuple[Callable, dict]],
    n_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process files through a pipeline of functions.

    Args:
        file_paths: List of file paths
        pipeline: List of (function, kwargs) tuples applied sequentially
        n_workers: Number of workers

    Returns:
        Dict mapping filepath -> final result
    """
    from .io import load_mzml, load_mzxml

    def _process(filepath):
        if filepath.lower().endswith(".mzxml"):
            result = load_mzxml(filepath)
        else:
            result = load_mzml(filepath)

        for func, kwargs in pipeline:
            result = func(result, **kwargs)
        return result

    processor = BatchProcessor(n_workers=n_workers)
    results = processor.process_files(file_paths, lambda fp: _process(fp))

    return {
        r.filepath: r.result
        for r in results
        if r.success
    }
