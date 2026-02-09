import os
import zipfile
import tarfile
import py7zr
import logging
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Extractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.extract_path = tempfile.mkdtemp()
        self.file_list = []

    def extract(self):
        """Extracts the archive based on extension."""
        try:
            if self.file_path.endswith('.zip'):
                self._extract_zip()
            elif self.file_path.endswith('.tar') or self.file_path.endswith('.tar.gz') or self.file_path.endswith('.tgz'):
                self._extract_tar()
            elif self.file_path.endswith('.7z'):
                self._extract_7z()
            else:
                raise ValueError(f"Unsupported file format: {self.file_path}")
            
            self._walk_files()
            return self.extract_path, self.file_list
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            self.cleanup()
            raise

    def _extract_zip(self):
        if not zipfile.is_zipfile(self.file_path):
            raise ValueError("Invalid zip file")
        with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
            # Check integrity
            bad_file = zip_ref.testzip()
            if bad_file:
                raise ValueError(f"Corrupted file in zip: {bad_file}")
            zip_ref.extractall(self.extract_path)

    def _extract_tar(self):
        if not tarfile.is_tarfile(self.file_path):
            raise ValueError("Invalid tar file")
        with tarfile.open(self.file_path, 'r:*') as tar_ref:
            tar_ref.extractall(self.extract_path)

    def _extract_7z(self):
        if not py7zr.is_7zfile(self.file_path):
            raise ValueError("Invalid 7z file")
        with py7zr.SevenZipFile(self.file_path, 'r') as archive:
            archive.extractall(self.extract_path)

    def _walk_files(self):
        """Walks through the extracted files and populates file_list."""
        for root, _, files in os.walk(self.extract_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.extract_path)
                self.file_list.append({
                    'full_path': full_path,
                    'rel_path': rel_path,
                    'size': os.path.getsize(full_path),
                    'extension': os.path.splitext(file)[1].lower()
                })

    def cleanup(self):
        """Cleans up the extracted files."""
        if os.path.exists(self.extract_path):
            shutil.rmtree(self.extract_path)
