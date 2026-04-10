"""Tests for the cloud module."""

import pytest
import os
from unittest.mock import MagicMock, patch

from masskit.cloud import (
    CloudConfig,
    DaskBackend,
    S3FileHandler,
    HPCJobSubmitter,
    generate_snakemake_workflow,
    generate_nextflow_workflow,
)


class TestCloudConfig:
    def test_defaults(self):
        config = CloudConfig()
        assert config.backend == "local"
        assert config.n_workers == 4
        assert config.memory_limit == "4GB"

    def test_custom(self):
        config = CloudConfig(
            backend="dask",
            n_workers=16,
            memory_limit="32GB",
            scheduler_address="tcp://1.2.3.4:8786",
        )
        assert config.n_workers == 16
        assert config.scheduler_address == "tcp://1.2.3.4:8786"


class TestDaskBackend:
    def test_init(self):
        backend = DaskBackend()
        assert backend.config is not None
        assert backend._client is None

    def test_init_with_config(self):
        config = CloudConfig(n_workers=2)
        backend = DaskBackend(config)
        assert backend.config.n_workers == 2

    def test_dashboard_link_when_closed(self):
        backend = DaskBackend()
        assert backend.dashboard_link is None

    def test_close_no_op(self):
        backend = DaskBackend()
        backend.close()  # Should not raise

    def test_start_with_mock_dask(self):
        backend = DaskBackend()
        mock_client = MagicMock()
        mock_client.dashboard_link = "http://localhost:8787"
        mock_cluster = MagicMock()

        with patch.dict("sys.modules", {
            "dask": MagicMock(),
            "dask.distributed": MagicMock(
                Client=MagicMock(return_value=mock_client),
                LocalCluster=MagicMock(return_value=mock_cluster),
            ),
        }):
            backend.start()
            assert backend._client is not None
            backend.close()


class TestS3FileHandler:
    def test_init(self):
        handler = S3FileHandler(bucket="test-bucket")
        assert handler.bucket == "test-bucket"
        assert handler._client is None

    def test_init_with_options(self):
        handler = S3FileHandler(
            bucket="b",
            endpoint_url="http://minio:9000",
            region="us-west-2",
            profile="test",
        )
        assert handler._endpoint_url == "http://minio:9000"
        assert handler._region == "us-west-2"

    def test_get_client_with_mock(self):
        handler = S3FileHandler(bucket="b")
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        with patch.dict("sys.modules", {
            "boto3": MagicMock(Session=MagicMock(return_value=mock_session)),
        }):
            client = handler._get_client()
            assert client is not None

    def test_list_files_with_mock(self):
        handler = S3FileHandler(bucket="b")
        handler._client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "raw/sample1.mzML"}, {"Key": "raw/sample2.txt"}]}
        ]
        handler._client.get_paginator.return_value = paginator

        files = handler.list_files(prefix="raw/", suffix=".mzML")
        assert "raw/sample1.mzML" in files
        assert "raw/sample2.txt" not in files

    def test_read_bytes_with_mock(self):
        handler = S3FileHandler(bucket="b")
        handler._client = MagicMock()
        handler._client.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b"test data"))
        }
        result = handler.read_bytes("key.txt")
        assert result == b"test data"

    def test_upload_file_with_mock(self, tmp_path):
        local = tmp_path / "test.txt"
        local.write_text("hello")
        handler = S3FileHandler(bucket="b")
        handler._client = MagicMock()
        url = handler.upload_file(str(local), "remote.txt")
        assert url == "s3://b/remote.txt"

    def test_upload_bytes_with_mock(self):
        handler = S3FileHandler(bucket="b")
        handler._client = MagicMock()
        url = handler.upload_bytes(b"data", "key.bin")
        assert url == "s3://b/key.bin"

    def test_stream_file(self):
        handler = S3FileHandler(bucket="b")
        handler._client = MagicMock()
        body_mock = MagicMock()
        body_mock.read.side_effect = [b"chunk1", b"chunk2", b""]
        handler._client.get_object.return_value = {"Body": body_mock}
        chunks = list(handler.stream_file("k", chunk_size=10))
        assert chunks == [b"chunk1", b"chunk2"]


class TestSnakemakeWorkflow:
    def test_generate(self, tmp_path):
        out = str(tmp_path / "Snakefile")
        path = generate_snakemake_workflow(output_path=out)
        assert path == out
        assert os.path.exists(out)
        content = open(out).read()
        assert "rule all" in content
        assert "rule peak_picking" in content

    def test_with_qc_step(self, tmp_path):
        out = str(tmp_path / "Snakefile")
        generate_snakemake_workflow(
            output_path=out,
            steps=["peak_picking", "feature_detection", "qc"],
        )
        content = open(out).read()
        assert "rule qc" in content

    def test_default_steps(self, tmp_path):
        out = str(tmp_path / "Snakefile")
        generate_snakemake_workflow(output_path=out)
        content = open(out).read()
        assert "consensus_map.csv" in content


class TestNextflowWorkflow:
    def test_generate(self, tmp_path):
        out = str(tmp_path / "main.nf")
        path = generate_nextflow_workflow(output_path=out)
        assert path == out
        content = open(out).read()
        assert "process peak_picking" in content
        assert "process quantification" in content


class TestHPCJobSubmitter:
    def test_invalid_scheduler(self):
        with pytest.raises(ValueError):
            HPCJobSubmitter(scheduler="lsf")

    def test_slurm_script(self):
        sub = HPCJobSubmitter(scheduler="slurm")
        script = sub.generate_script(
            "masskit info test.mzML",
            job_name="test",
            cpus=8,
            memory="16G",
            time="4:00:00",
            partition="general",
        )
        assert "#SBATCH --job-name=test" in script
        assert "#SBATCH --cpus-per-task=8" in script
        assert "#SBATCH --mem=16G" in script
        assert "#SBATCH --partition=general" in script

    def test_pbs_script(self):
        sub = HPCJobSubmitter(scheduler="pbs")
        script = sub.generate_script(
            "masskit qc *.mzML",
            cpus=4,
        )
        assert "#PBS -N masskit" in script
        assert "#PBS -l ncpus=4" in script

    def test_extra_directives(self):
        sub = HPCJobSubmitter(scheduler="slurm")
        script = sub.generate_script(
            "echo hello",
            extra_directives=["--gres=gpu:1"],
        )
        assert "--gres=gpu:1" in script

    def test_write_script(self, tmp_path):
        sub = HPCJobSubmitter(scheduler="slurm")
        script = sub.generate_script("echo test")
        out = str(tmp_path / "job.sh")
        sub.write_script(script, out)
        assert os.path.exists(out)
        # Verify executable bit
        assert os.access(out, os.X_OK)

    def test_array_script_slurm(self, tmp_path):
        sub = HPCJobSubmitter(scheduler="slurm")
        files = ["a.mzML", "b.mzML", "c.mzML"]
        filelist = str(tmp_path / "files.txt")
        script = sub.generate_array_script(
            files,
            filelist_path=filelist,
        )
        assert "--array=1-3" in script
        assert os.path.exists(filelist)

    def test_array_script_pbs(self, tmp_path):
        sub = HPCJobSubmitter(scheduler="pbs")
        files = ["a.mzML", "b.mzML"]
        filelist = str(tmp_path / "files.txt")
        script = sub.generate_array_script(
            files,
            filelist_path=filelist,
        )
        assert "-J 1-2" in script
