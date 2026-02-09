import os
import pandas as pd
import json
import logging
import chardet
import numpy as np
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class FileParser:
    def __init__(self, file_info):
        self.file_path = file_info['full_path']
        self.rel_path = file_info['rel_path']
        self.extension = file_info['extension']
        self.size = file_info['size']
        self.summary = {
            'file_name': self.rel_path,
            'file_type': self.extension,
            'size': self.size,
            'record_count': 0,
            'field_count': 0,
            'columns': [],
            'missing_ratio': 0,
            'duplicate_count': 0,
            'sample_data': [],
            'error': None
        }
        self.df = None

    def parse(self):
        try:
            if self.extension == '.csv':
                self._parse_csv()
            elif self.extension == '.json':
                self._parse_json()
            elif self.extension in ['.xls', '.xlsx']:
                self._parse_excel()
            elif self.extension == '.xml':
                self._parse_xml()
            elif self.extension == '.txt':
                self._parse_txt()
            else:
                self.summary['file_type'] = 'Binary/Other'
                # Binary files - just return basic info
                pass
            
            if self.df is not None:
                self._analyze_dataframe()
                
        except Exception as e:
            logger.error(f"Error parsing {self.file_path}: {e}")
            self.summary['error'] = str(e)
        
        return self.summary

    def _detect_encoding(self):
        with open(self.file_path, 'rb') as f:
            raw = f.read(10000)
        return chardet.detect(raw)['encoding']

    def _parse_csv(self):
        encoding = self._detect_encoding()
        # Handle large files with chunking if size > 100MB
        if self.size > 100 * 1024 * 1024:
            self._parse_large_csv(encoding)
        else:
            self.df = pd.read_csv(self.file_path, encoding=encoding)

    def _parse_large_csv(self, encoding):
        # Process in chunks, but for summary we'll just take a sample or aggregate stats
        # For simplicity in this demo, we read first chunk for sample/schema and count lines
        chunk_size = 10000
        chunks = pd.read_csv(self.file_path, encoding=encoding, chunksize=chunk_size)
        self.df = next(chunks) # Use first chunk for deep analysis preview
        
        # Count total records roughly
        count = 0
        for chunk in pd.read_csv(self.file_path, encoding=encoding, chunksize=chunk_size, usecols=[0]):
             count += len(chunk)
        self.summary['record_count'] = count
        self.summary['is_large_file'] = True

    def _parse_json(self):
        encoding = self._detect_encoding()
        try:
            self.df = pd.read_json(self.file_path, encoding=encoding)
        except ValueError:
            # Maybe it's line-delimited JSON
            self.df = pd.read_json(self.file_path, encoding=encoding, lines=True)

    def _parse_excel(self):
        self.df = pd.read_excel(self.file_path)

    def _parse_xml(self):
        # Simple XML to DataFrame (flat structure assumption)
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        data = []
        for child in root:
            data.append({elem.tag: elem.text for elem in child})
        self.df = pd.DataFrame(data)

    def _parse_txt(self):
        encoding = self._detect_encoding()
        with open(self.file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
        self.summary['record_count'] = len(lines)
        self.summary['sample_data'] = lines[:20]
        # Treat as single column DF for consistency if needed, or just leave as is
        self.df = pd.DataFrame(lines, columns=['content'])

    def _analyze_dataframe(self):
        if self.df is None or self.df.empty:
            return

        # Basic Stats
        if 'record_count' not in self.summary or self.summary['record_count'] == 0:
            self.summary['record_count'] = len(self.df)
        
        self.summary['field_count'] = len(self.df.columns)
        self.summary['columns'] = list(self.df.columns)
        
        # Data Types
        dtypes = self.df.dtypes.astype(str).to_dict()
        self.summary['dtypes'] = dtypes
        
        # Missing Values
        missing_total = self.df.isnull().sum().sum()
        total_cells = self.df.size
        self.summary['missing_ratio'] = missing_total / total_cells if total_cells > 0 else 0
        
        # Duplicates
        self.summary['duplicate_count'] = self.df.duplicated().sum()
        
        # Field Stats (for numerical)
        field_stats = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            desc = self.df[col].describe()
            field_stats[col] = {
                'mean': desc['mean'],
                'std': desc['std'],
                'min': desc['min'],
                'max': desc['max'],
                'median': self.df[col].median()
            }
            # Outliers (IQR)
            Q1 = desc['25%']
            Q3 = desc['75%']
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            field_stats[col]['outliers'] = int(outliers)
            
        self.summary['field_stats'] = field_stats

        # Sample Data (Desensitize)
        sample = self.df.head(20).copy()
        sample = self._desensitize(sample)
        # Convert to dict for JSON serialization, handling NaN/Infinity
        self.summary['sample_data'] = sample.astype(object).where(pd.notnull(sample), None).to_dict(orient='records')

    def _desensitize(self, df):
        # Simple desensitization for email and phone-like patterns
        # This is a basic implementation
        for col in df.columns:
            if df[col].dtype == 'object':
                # Mask emails
                df[col] = df[col].astype(str).str.replace(r'(\w{2})[\w.-]+@([\w.-]+)', r'\1***@\2', regex=True)
                # Mask phones (simple 10+ digits)
                df[col] = df[col].astype(str).str.replace(r'\d{3}\d{4}(\d{4})', r'***-****-\1', regex=True)
        return df
