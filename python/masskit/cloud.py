"""
Cloud and HPC integration for MassKit.

Provides Dask backend for distributed processing, S3 streaming,
and Snakemake/Nextflow workflow template generation.
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import os
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    """Configuration for cloud/HPC execution."""
    backend: str = "local"  # 'local', 'dask', 'dask-distributed'
    n_workers: int = 4
    memory_limit: str = "4GB"
    scheduler_address: Optional[str] = None
    s3_endpoint: Optional[str] = None
    s3_region: Optional[str] = None
    temp_dir: str = "/tmp/masskit"


class DaskBackend:
    """
    Dask-based distributed processing backend.

    Provides transparent parallelization of LC-MS processing tasks
    across local threads, processes, or distributed clusters.

    Example:
        >>> backend = DaskBackend(n_workers=8)
        >>> results = backend.map(process_file, file_list)
        >>> backend.close()
    """

    def __init__(self, config: Optional[CloudConfig] = None):
        self.config = config or CloudConfig()
        self._client = None
        self._cluster = None

    def start(self) -> None:
        """Start the Dask cluster/client."""
        try:
            import dask
            from dask.distributed import Client, LocalCluster
        except ImportError:
            raise ImportError("Dask is required: pip install dask[distributed]")

        if self.config.scheduler_address:
            self._client = Client(self.config.scheduler_address)
        else:
            self._cluster = LocalCluster(
                n_workers=self.config.n_workers,
                memory_limit=self.config.memory_limit,
            )
            self._client = Client(self._cluster)

        logger.info(f"Dask client started: {self._client.dashboard_link}")

    @property
    def client(self):
        if self._client is None:
            self.start()
        return self._client

    def map(self, func: Callable, items: List[Any], **kwargs) -> List[Any]:
        """Map a function over items using Dask."""
        futures = self.client.map(func, items, **kwargs)
        return self.client.gather(futures)

    def submit(self, func: Callable, *args, **kwargs):
        """Submit a single task."""
        return self.client.submit(func, *args, **kwargs)

    def scatter(self, data: Any):
        """Scatter data to workers."""
        return self.client.scatter(data)

    @property
    def dashboard_link(self) -> Optional[str]:
        if self._client:
            return self._client.dashboard_link
        return None

    def close(self) -> None:
        """Shut down the Dask client and cluster."""
        if self._client:
            self._client.close()
            self._client = None
        if self._cluster:
            self._cluster.close()
            self._cluster = None


class S3FileHandler:
    """
    S3-compatible file streaming for LC-MS data.

    Supports reading/writing mzML and other files from S3-compatible
    storage (AWS S3, MinIO, etc.).

    Example:
        >>> s3 = S3FileHandler(bucket="my-lcms-data")
        >>> data = s3.read_bytes("raw/sample1.mzML")
        >>> s3.upload_file("local_results.csv", "results/output.csv")
    """

    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
        profile: Optional[str] = None,
    ):
        self.bucket = bucket
        self._endpoint_url = endpoint_url
        self._region = region
        self._profile = profile
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 is required: pip install boto3")

            session_kwargs = {}
            if self._profile:
                session_kwargs["profile_name"] = self._profile

            session = boto3.Session(**session_kwargs)
            client_kwargs = {}
            if self._endpoint_url:
                client_kwargs["endpoint_url"] = self._endpoint_url
            if self._region:
                client_kwargs["region_name"] = self._region

            self._client = session.client("s3", **client_kwargs)
        return self._client

    def list_files(self, prefix: str = "", suffix: str = "") -> List[str]:
        """List files in the bucket with optional prefix/suffix filter."""
        client = self._get_client()
        paginator = client.get_paginator("list_objects_v2")
        files = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if suffix and not key.endswith(suffix):
                    continue
                files.append(key)
        return files

    def read_bytes(self, key: str) -> bytes:
        """Read a file from S3 as bytes."""
        client = self._get_client()
        response = client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def download_file(self, key: str, local_path: str) -> str:
        """Download a file from S3 to local path."""
        client = self._get_client()
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        client.download_file(self.bucket, key, local_path)
        return local_path

    def upload_file(self, local_path: str, key: str) -> str:
        """Upload a local file to S3."""
        client = self._get_client()
        client.upload_file(local_path, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def upload_bytes(self, data: bytes, key: str) -> str:
        """Upload bytes to S3."""
        client = self._get_client()
        client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    def stream_file(self, key: str, chunk_size: int = 8 * 1024 * 1024):
        """Stream a file from S3 in chunks (generator)."""
        client = self._get_client()
        response = client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"]
        while True:
            chunk = body.read(chunk_size)
            if not chunk:
                break
            yield chunk


def generate_snakemake_workflow(
    input_dir: str = "raw/",
    output_dir: str = "results/",
    steps: Optional[List[str]] = None,
    output_path: str = "Snakefile",
) -> str:
    """
    Generate a Snakemake workflow template for LC-MS analysis.

    Args:
        input_dir: Input data directory
        output_dir: Output results directory
        steps: Processing steps to include
        output_path: Output Snakefile path

    Returns:
        Path to generated Snakefile
    """
    if steps is None:
        steps = ["peak_picking", "feature_detection", "alignment", "quantification"]

    snakefile = f'''# MassKit Snakemake Workflow
# Generated by masskit.cloud

import glob

INPUT_DIR = "{input_dir}"
OUTPUT_DIR = "{output_dir}"

SAMPLES = glob_wildcards(INPUT_DIR + "{{sample}}.mzML").sample

rule all:
    input:
        expand(OUTPUT_DIR + "{{sample}}.peaks.csv", sample=SAMPLES),
'''

    if "feature_detection" in steps:
        snakefile += '        expand(OUTPUT_DIR + "{sample}.features.csv", sample=SAMPLES),\n'
    if "alignment" in steps or "quantification" in steps:
        snakefile += f'        OUTPUT_DIR + "consensus_map.csv",\n'

    snakefile += f'''
rule peak_picking:
    input:
        INPUT_DIR + "{{sample}}.mzML"
    output:
        OUTPUT_DIR + "{{sample}}.peaks.csv"
    shell:
        "masskit peaks {{input}} -o {{output}} --snr 3.0"
'''

    if "feature_detection" in steps:
        snakefile += f'''
rule feature_detection:
    input:
        INPUT_DIR + "{{sample}}.mzML"
    output:
        OUTPUT_DIR + "{{sample}}.features.csv"
    shell:
        "masskit peaks {{input}} -o {{output}} --snr 3.0"
'''

    if "alignment" in steps or "quantification" in steps:
        snakefile += f'''
rule quantification:
    input:
        expand(OUTPUT_DIR + "{{sample}}.features.csv", sample=SAMPLES)
    output:
        OUTPUT_DIR + "consensus_map.csv"
    shell:
        "masskit quantify {{input}} -o {{output}}"
'''

    if "qc" in steps:
        snakefile += f'''
rule qc:
    input:
        INPUT_DIR + "{{sample}}.mzML"
    output:
        OUTPUT_DIR + "{{sample}}.qc.json"
    shell:
        "masskit qc {{input}} -o {{output}}"
'''

    Path(output_path).write_text(snakefile)
    return output_path


def generate_nextflow_workflow(
    output_path: str = "main.nf",
) -> str:
    """
    Generate a Nextflow workflow template for LC-MS analysis.

    Args:
        output_path: Output .nf file path

    Returns:
        Path to generated workflow file
    """
    nf_content = '''#!/usr/bin/env nextflow
// MassKit Nextflow Workflow
// Generated by masskit.cloud

params.input_dir = "raw/"
params.output_dir = "results/"
params.snr = 3.0

Channel
    .fromPath("${params.input_dir}/*.mzML")
    .set { mzml_files }

process peak_picking {
    publishDir params.output_dir, mode: 'copy'

    input:
    path mzml from mzml_files

    output:
    path "${mzml.baseName}.peaks.csv" into peaks_ch

    script:
    """
    masskit peaks ${mzml} -o ${mzml.baseName}.peaks.csv --snr ${params.snr}
    """
}

process quantification {
    publishDir params.output_dir, mode: 'copy'

    input:
    path peaks from peaks_ch.collect()

    output:
    path "consensus_map.csv"

    script:
    """
    masskit quantify ${peaks} -o consensus_map.csv
    """
}
'''

    Path(output_path).write_text(nf_content)
    return output_path


class HPCJobSubmitter:
    """
    Submit LC-MS processing jobs to HPC schedulers (SLURM, PBS).

    Example:
        >>> submitter = HPCJobSubmitter(scheduler="slurm")
        >>> job_id = submitter.submit("masskit peaks sample.mzML -o peaks.csv",
        ...                           cpus=4, memory="8G", time="1:00:00")
    """

    def __init__(self, scheduler: str = "slurm"):
        if scheduler not in ("slurm", "pbs"):
            raise ValueError(f"Unsupported scheduler: {scheduler}")
        self.scheduler = scheduler

    def generate_script(
        self,
        command: str,
        job_name: str = "masskit",
        cpus: int = 4,
        memory: str = "8G",
        time: str = "2:00:00",
        partition: Optional[str] = None,
        output_log: str = "masskit_%j.out",
        extra_directives: Optional[List[str]] = None,
    ) -> str:
        """Generate a job submission script."""
        if self.scheduler == "slurm":
            lines = ["#!/bin/bash"]
            lines.append(f"#SBATCH --job-name={job_name}")
            lines.append(f"#SBATCH --cpus-per-task={cpus}")
            lines.append(f"#SBATCH --mem={memory}")
            lines.append(f"#SBATCH --time={time}")
            lines.append(f"#SBATCH --output={output_log}")
            if partition:
                lines.append(f"#SBATCH --partition={partition}")
            if extra_directives:
                for d in extra_directives:
                    lines.append(f"#SBATCH {d}")
            lines.append("")
            lines.append("module load python 2>/dev/null || true")
            lines.append(command)
        else:  # PBS
            lines = ["#!/bin/bash"]
            lines.append(f"#PBS -N {job_name}")
            lines.append(f"#PBS -l ncpus={cpus}")
            lines.append(f"#PBS -l mem={memory}")
            lines.append(f"#PBS -l walltime={time}")
            lines.append(f"#PBS -o {output_log}")
            if partition:
                lines.append(f"#PBS -q {partition}")
            if extra_directives:
                for d in extra_directives:
                    lines.append(f"#PBS {d}")
            lines.append("")
            lines.append("module load python 2>/dev/null || true")
            lines.append(f"cd $PBS_O_WORKDIR")
            lines.append(command)

        return "\n".join(lines) + "\n"

    def write_script(self, script: str, filepath: str) -> str:
        """Write a job script to file."""
        Path(filepath).write_text(script)
        os.chmod(filepath, 0o755)
        return filepath

    def generate_array_script(
        self,
        files: List[str],
        command_template: str = "masskit peaks {file} -o {output}",
        output_dir: str = "results/",
        **kwargs,
    ) -> str:
        """
        Generate an array job script for processing multiple files.

        Args:
            files: List of input files
            command_template: Command with {file} and {output} placeholders
            output_dir: Output directory
            **kwargs: Additional job parameters
        """
        filelist_path = kwargs.pop("filelist_path", "filelist.txt")
        Path(filelist_path).write_text("\n".join(files))

        if self.scheduler == "slurm":
            array_cmd = (
                f"FILE=$(sed -n \"${{SLURM_ARRAY_TASK_ID}}p\" {filelist_path})\n"
                f"BASENAME=$(basename \"$FILE\" .mzML)\n"
                f"OUTPUT=\"{output_dir}/${{BASENAME}}.csv\"\n"
            )
            cmd = command_template.replace("{file}", "$FILE").replace("{output}", "$OUTPUT")
            script = self.generate_script(
                array_cmd + cmd,
                job_name=kwargs.get("job_name", "masskit_array"),
                cpus=kwargs.get("cpus", 4),
                memory=kwargs.get("memory", "8G"),
                time=kwargs.get("time", "2:00:00"),
                extra_directives=[f"--array=1-{len(files)}"],
            )
        else:  # PBS
            array_cmd = (
                f"FILE=$(sed -n \"${{PBS_ARRAY_INDEX}}p\" {filelist_path})\n"
                f"BASENAME=$(basename \"$FILE\" .mzML)\n"
                f"OUTPUT=\"{output_dir}/${{BASENAME}}.csv\"\n"
            )
            cmd = command_template.replace("{file}", "$FILE").replace("{output}", "$OUTPUT")
            script = self.generate_script(
                array_cmd + cmd,
                job_name=kwargs.get("job_name", "masskit_array"),
                cpus=kwargs.get("cpus", 4),
                memory=kwargs.get("memory", "8G"),
                time=kwargs.get("time", "2:00:00"),
                extra_directives=[f"-J 1-{len(files)}"],
            )

        return script
