import json
import os
from jinja2 import Environment, FileSystemLoader
import plotly.express as px
import plotly.utils
import pandas as pd

class Reporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.results = []
        self.summary = {
            'total_files': 0,
            'total_size': 0,
            'files': []
        }

    def add_result(self, result):
        self.results.append(result)
        self.summary['total_files'] += 1
        self.summary['total_size'] += result['size']
        self.summary['files'].append(result)

    def generate_json(self):
        output_path = os.path.join(self.output_dir, 'analysis_summary.json')
        # Handle NaN/Infinity for JSON dump
        def safe_serialize(obj):
            if isinstance(obj, float):
                 if obj != obj: return None # NaN
                 if obj == float('inf') or obj == float('-inf'): return None
            return obj

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=4, default=str)
        return output_path

    def generate_html(self):
        output_path = os.path.join(self.output_dir, 'analysis_report.html')
        
        # Prepare Chart Data
        df_files = pd.DataFrame(self.summary['files'])
        if not df_files.empty:
            fig = px.pie(df_files, names='file_type', title='File Type Distribution')
            file_type_chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            file_type_chart_json = "{data: [], layout: {}}"

        env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
        template = env.get_template('report_template.html')
        
        # Add derived stats
        self.summary['total_size_mb'] = round(self.summary['total_size'] / (1024 * 1024), 2)

        html_content = template.render(
            summary=self.summary,
            file_type_chart_json=file_type_chart_json
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
