import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from google.api_core.exceptions import Forbidden
from google.cloud import storage
from tqdm import tqdm
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_file(url, destination):
    """Download a file from a direct URL to the local destination."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
    
    return destination


def download_gcs_directory(
    bucket_name: str,
    destination_dir: str,
    prefix: str = "",
    max_workers: int = 8,
) -> None:
    """
    Downloads files from a GCS 'directory' (prefix) in parallel.

    Args:
        bucket_name: The name of the GCS bucket.
        destination_dir: The local directory to save files.
        prefix: The prefix to filter blobs (acts like a directory path).
        max_workers: The maximum number of parallel download threads.
    """
    os.makedirs(destination_dir, exist_ok=True)

    client: storage.Client
    try:
        # Try authenticated client first (uses ADC)
        logging.info("Attempting authenticated GCS access...")
        client = storage.Client()
        # Perform a test operation to check credentials
        client.list_buckets(max_results=1)
        logging.info("Authenticated GCS access successful.")
    except Exception as auth_ex:
        logging.warning(
            f"Authenticated access failed ({auth_ex}). Attempting anonymous access..."
        )
        try:
            # Fallback to anonymous client
            client = storage.Client.create_anonymous_client()
            # Test anonymous access
            bucket = client.bucket(bucket_name)
            bucket.exists() # Check if bucket is accessible
            logging.info("Anonymous GCS access successful.")
        except Forbidden:
             logging.error(
                "Anonymous access denied. Please ensure you are authenticated. "
                "Run 'gcloud auth application-default login --no-launch-browser'"
             )
             return
        except Exception as anon_ex:
            logging.error(f"Failed to initialize anonymous GCS client: {anon_ex}")
            return

    logging.info(f"Listing files in gs://{bucket_name}/{prefix}...")
    try:
        blobs_to_download: List[str] = [
            blob.name for blob in client.list_blobs(bucket_name, prefix=prefix)
            if not blob.name.endswith('/') # Exclude 'directory' placeholders
        ]
    except Exception as e:
        logging.error(f"Failed to list blobs in bucket {bucket_name} with prefix '{prefix}': {e}")
        return


    if not blobs_to_download:
        logging.warning(f"No files found in gs://{bucket_name}/{prefix}")
        return

    logging.info(f"Found {len(blobs_to_download)} files to potentially download.")

    successful_downloads = 0
    skipped_downloads = 0
    failed_downloads = 0

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create future tasks
        future_to_blob = {
            executor.submit(
                download_file, blob_name, os.path.join(destination_dir, os.path.basename(blob_name))
            ): blob_name
            for blob_name in blobs_to_download
        }

        # Process completed futures with a progress bar
        for future in tqdm(
            as_completed(future_to_blob),
            total=len(blobs_to_download),
            desc="Downloading files",
            unit="file",
        ):
            blob_name = future_to_blob[future]
            try:
                result = future.result()
                if result is not None: # Successfully downloaded
                     successful_downloads += 1
                else: # Skipped (already exists) or failed
                     # Check logs inside download_file for failure reason
                     # We count explicit skips vs failures if needed by checking logs/return codes
                     # For simplicity here, we assume None means skip/fail
                     # Refined check: could return specific codes from download_file
                     destination_file_name = result
                     if os.path.exists(destination_file_name): # Assume skip if exists post-call
                         skipped_downloads +=1
                         logging.info(f"Skipped existing file: {blob_name}")
                     else: # Assume failure if None returned and file doesn't exist
                         failed_downloads += 1
                         # Error already logged in download_file

            except Exception as exc:
                logging.error(f"Download task for {blob_name} generated an exception: {exc}")
                failed_downloads += 1

    logging.info("-" * 30)
    logging.info("Download Summary:")
    logging.info(f"  Successfully downloaded: {successful_downloads}")
    logging.info(f"  Skipped (already exist): {skipped_downloads}")
    logging.info(f"  Failed: {failed_downloads}")
    logging.info("-" * 30)
    logging.info(f"Downloads saved to: {os.path.abspath(destination_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download files from a Google Cloud Storage bucket prefix."
    )
    parser.add_argument(
        "--bucket_name",
        type=str,
        required=True,
        help="Name of the GCS bucket.",
        default="waymo_open_dataset_end_to_end_camera_v_1_0_0" # Default based on user context
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix (folder path) within the bucket to download from. Leave empty to download all.",
    )
    parser.add_argument(
        "--destination_dir",
        type=str,
        required=True,
        help="Local directory to save downloaded files.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8, # Adjust based on cluster network/CPU
        help="Maximum number of parallel download threads.",
    )

    args = parser.parse_args()

    # Example: Download the test split files based on user context
    # python download.py \
    #   --bucket_name waymo_open_dataset_end_to_end_camera_v_1_0_0 \
    #   --prefix "test_" \
    #   --destination_dir /path/to/your/data/waymo_e2e_camera/test \
    #   --max_workers 16

    download_gcs_directory(
        bucket_name=args.bucket_name,
        destination_dir=args.destination_dir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )
